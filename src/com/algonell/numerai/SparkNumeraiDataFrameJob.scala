package com.algonell.numerai

import breeze.linalg.{max, min}
import breeze.numerics.log
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vectors, DenseVector}
import org.apache.spark.sql._
import org.apache.spark.sql.types.DoubleType

/**
  * Created by andrewkreimer on 6/28/16.
  */
object SparkNumeraiDataFrameJob {

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //configure Spark
    implicit val spark = SparkSession.builder().appName("Numerai").master("local[*]").getOrCreate()

    //get data
    val (cols, trainData) = getTrainData
    val testData = getTestData(cols)

    // split data
    val (train: Dataset[Row], test: Dataset[Row]) = split(trainData)

    //LR
    println("LR:")
    val lr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(100)
    val lrModel = lr.fit(train)
    val lrPredictions = lrModel.transform(test)
    lrPredictions.show(5)
    calculateLogLoss(lrPredictions)

    //RF
    println("RF:")
    val rf = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(100)
    val rfModel = rf.fit(train)
    var rfPredictions = rfModel.transform(test)
    rfPredictions.show(5)
    calculateLogLoss(rfPredictions)

    /*
    //GBC
    println("GBC:")
    val gbc = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(100)
    val gbcModel = gbc.fit(train)
    val gbcPredictions = gbcModel.transform(test)
    gbcPredictions.show(5)
    calculateLogLoss(spark, gbcPredictions)
    */

    //NB
    println("NB:")
    val nb = new NaiveBayes().setLabelCol("label").setFeaturesCol("features")
    val nbModel = nb.fit(train)
    val nbPredictions = nbModel.transform(test)
    nbPredictions.show(5)
    calculateLogLoss(nbPredictions)
    nbPredictions.show()

    //Stacking
    val dfToCombine = Map("LR" -> rfPredictions, "RF" -> lrPredictions, "NB" -> nbPredictions)
    val dfBeforeJoin = dfToCombine.map(entry => entry._1 -> entry._2.select("label", "features", "probability"))
    val dfRenamedProbability = dfBeforeJoin.map(entry => entry._2.withColumnRenamed("probability", entry._1))

    dfRenamedProbability.foreach(df => println("Before:" + df.count()))
    dfRenamedProbability.foreach(_.show)

    var stackedDf = dfRenamedProbability.reduce{
      (df1 : DataFrame, df2: DataFrame) => df1.join(df2, Seq("label","features"))
    }

    println("After:" + stackedDf.count())

    stackedDf = stackedDf.drop(stackedDf.col("features"))
    stackedDf.show()

    //flat the stacked data
    stackedDf = new VectorAssembler().setInputCols(dfToCombine.keySet.toArray).setOutputCol("features").transform(stackedDf)
    stackedDf.show()

    stackedDf.select("features").take(5).foreach(println)

    import spark.implicits._

    implicit val myObjEncoder = org.apache.spark.sql.Encoders.kryo[org.apache.spark.ml.linalg.Vector]

    //unbox probabilities for both labels
    val df = stackedDf.map{row =>
      val label = row.getAs[Double](0)
      val lr = row.get(1).asInstanceOf[DenseVector]
      val rf = row.get(2).asInstanceOf[DenseVector]
      val nb = row.get(3).asInstanceOf[DenseVector]

      Vectors.dense(label)
    }

    //show stacked data
    stackedDf = stackedDf.drop(stackedDf.col("features"))
    stackedDf.show()

    //flat the stacked data
    stackedDf = new VectorAssembler().setInputCols(dfToCombine.keySet.toArray).setOutputCol("features").transform(stackedDf)

    dfToCombine.keySet.map(key =>
      stackedDf = stackedDf.drop(stackedDf.col(key))
    )

    //show flatted data
    stackedDf.show()

    //Stacked LR
    println("LR:")
    val (stackedTrain: Dataset[Row], stackedTest: Dataset[Row]) = split(stackedDf)
    val stackedLr = new LogisticRegression().setFeaturesCol("features").setLabelCol("label").setMaxIter(100)
    val stackedLrModel = stackedLr.fit(stackedTrain)
    var stackedLrPredictions = stackedLrModel.transform(stackedTest)
    stackedLrPredictions.show(5)
    calculateLogLoss(stackedLrPredictions)

    /*
    //make submission
    val submission = gbcModel.transform(testData)

    //write submission file
    val created : String = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm").format(new Date())
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/submissions/" + created + "_spark_GBC.csv"))
    pw.write("t_id,probability\n")
    submission.collect().foreach(row =>
      pw.write(String.valueOf(row.get(0)))
    )
    pw.close()
    */
  }

  /**
    * Split for validation
    *
    * @param trainData - entire data set
    * @return
    */
  def split(trainData: DataFrame): (Dataset[Row], Dataset[Row]) = {
    println("Splitting data...")
    val split = 0.8
    val splits = trainData.randomSplit(Array(split, 1 - split), seed = 11L)

    val train = splits(0).cache()
    val test = splits(1)

    (train, test)
  }

  /**
    * Get train data
    *
    * @return
    */
  def getTrainData(implicit spark : SparkSession): (Array[String], DataFrame) = {
    //load train data
    var trainData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_training_data.csv")
    trainData = trainData.withColumn("target", trainData("target").cast(DoubleType))

    //convert to LabeledPoint structure
    val cols = Array.ofDim[String](trainData.columns.length - 1)
    Array.copy(trainData.columns, 0, cols, 0, trainData.columns.length - 1)
    trainData = new VectorAssembler().setInputCols(cols).setOutputCol("features").transform(trainData)

    //drop unused cols
    cols.foreach(col =>
      trainData = trainData.drop(col)
    )

    //rename class variable
    trainData = trainData.withColumnRenamed("target", "label")

    //show train
    println("Train Data:")
    trainData.show()

    (cols, trainData)
  }

  /**
    * Get test data, target is id
    */
  def getTestData(cols : Array[String])(implicit spark : SparkSession) : DataFrame = {
    //load test data
    var testData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_tournament_data.csv")

    //convert to LabeledPoint structure
    testData = new VectorAssembler().setInputCols(cols).setOutputCol("features").transform(testData)

    //drop unused cols
    cols.foreach(col =>
      testData = testData.drop(col)
    )

    //show test
    println("Test Data:")
    testData.show()

    testData
  }

  /**
    * Calculate log loss
 *
    * @param predictions - result of model transformation
    */
  def calculateLogLoss(predictions: DataFrame)(implicit spark : SparkSession): Unit = {
    //log loss
    val loss = spark.sparkContext.doubleAccumulator("LogLoss")
    val trainCount = spark.sparkContext.longAccumulator("Global train examples count")
    val maxmin = (p: Double) => max(min(p, 1.0 - 1e-14), 1e-14)
    val logloss: ((Double, Double) => Double) = (p: Double, y: Double) => -(y * log(p) + (1 - y) * log(1 - p))

    // Check log loss on training dataset
    val predictionAndlabel = predictions.select("probability", "label")

    predictionAndlabel.foreach(row => {
      loss.add(logloss(row.getAs[DenseVector](0).toArray.last, row.getAs[Double](1)))
      trainCount.add(1)
    })

    val totalLoss = loss.value / trainCount.value
    println("totalLoss: " + totalLoss)
  }
}