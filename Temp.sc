import breeze.linalg.{max, min}
import breeze.numerics.log
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql._

//configure Spark
val spark = SparkSession.builder().appName("Spark SQL Example").master("local[*]").getOrCreate()
/**
  * Get train data
  *
  * @return
  */
def getTrainData(spark : SparkSession): (Array[String], DataFrame) = {
  //load train data
  var trainData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_training_data.csv")
  println("Raw Data:")
  trainData.show()
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
def getTestData(spark : SparkSession, cols : Array[String]): DataFrame = {
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
  * @param spark
  * @param predictions
  */
def calculateLogLoss(spark: SparkSession, predictions: DataFrame): Unit = {
  //log loss
  val loss = spark.sparkContext.doubleAccumulator("LogLoss")
  val trainCount = spark.sparkContext.longAccumulator("Global train examples count")
  val maxmin = (p: Double) => max(min(p, 1.0 - 1e-14), 1e-14)
  val logloss: ((Double, Integer) => Double) = (p: Double, y: Integer) => -(y * log(p) + (1 - y) * log(1 - p))

  // Check log loss on training dataset
  val predictionAndlabel = predictions.select("probability", "label")

  predictionAndlabel.foreach(row => {
    loss.add(logloss(row.getAs[DenseVector](0).toArray.last, row.getAs[Integer](1)))
    trainCount.add(1)
  })

  val totalLoss = loss.value / trainCount.value
  println("totalLoss: " + totalLoss)
}

//get data
val (cols, trainData) = getTrainData(spark)
val testData = getTestData(spark, cols)