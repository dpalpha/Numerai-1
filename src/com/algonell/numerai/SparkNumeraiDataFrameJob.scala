package com.algonell.numerai

import java.io.{File, PrintWriter}

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorSlicer}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by andrewkreimer on 6/28/16.
  */
object SparkNumeraiDataFrameJob {
  def main(args: Array[String]) {
    //configure Spark
    val spark = SparkSession.builder().appName("Spark SQL Example").master("local[*]").getOrCreate()

    //load train data
    var trainData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_training_data.csv")

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
    trainData.show()

    //load test data
    var testData = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_tournament_data.csv")

    //convert to LabeledPoint structure
    testData = new VectorAssembler().setInputCols(cols).setOutputCol("features").transform(testData)

    //drop unused cols
    cols.foreach(col =>
      testData = testData.drop(col)
    )

    //show test
    testData.show()

    // split data
    val split = 0.9
    val splits = trainData.randomSplit(Array(split, 1 - split), seed = 11L)

    val train = splits(0).cache()
    val test = splits(1)

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val model = lr.fit(train)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // Compute raw scores on the test set
    val predictions = model.transform(test)

    // Instantiate metrics object
    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
  }

  //convert df to numbers
  def toNumbers(df: DataFrame):DataFrame = {
    df.columns.foreach(col =>
      df.withColumn(col, df.col(col).cast(DoubleType))
      .drop(col)
      .withColumnRenamed(col, col)
    )

    df
  }

  def trainLRModel(testData: RDD[LabeledPoint], train: RDD[LabeledPoint], test: RDD[LabeledPoint]) = {
    // train LR model
    val lrAlgorithm = new LogisticRegressionWithLBFGS()
    lrAlgorithm.setNumClasses(2)
    lrAlgorithm.optimizer.setNumIterations(100000)
    lrAlgorithm.optimizer.setRegParam(0.1)
    lrAlgorithm.optimizer.setConvergenceTol(0.001)
    lrAlgorithm.optimizer.setNumCorrections(100)
    val lrModel = lrAlgorithm.run(train)

    // evaluate
    val predictionAndLabels = test.map { point =>
      val prediction = lrModel.predict(point.features)
      (point.label, prediction)
    }

    // precision/recall
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("LR: Precision: " + metrics.precision + " Recall: " + metrics.recall)

    //extract probabilities
    lrModel.clearThreshold()

    //make submission
    val submission = testData.map { point =>
      val prediction = lrModel.predict(point.features)
      (point.label, prediction)
    }

    //write submission file
    val collect: Array[(Double, Double)] = submission.collect()
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/submissions/2016_07_03_spark_LR.csv"))
    pw.write("t_id,probability\n")
    collect.foreach(tuple => pw.write(tuple._1.toInt + "," + tuple._2 + "\n"))
    pw.close()
  }

  def trainGBCModel(testData: RDD[LabeledPoint], train: RDD[LabeledPoint], test: RDD[LabeledPoint]) = {
    // train GBC model
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(3) // Note: Use more iterations in practice.
    //oostingStrategy.treeStrategy.numClasses = 2
    //boostingStrategy.treeStrategy.maxDepth = 5
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    val gbcmModel = GradientBoostedTrees.train(train, boostingStrategy)

    // evaluate
    val predictionAndLabels = test.map { point =>
      val prediction = gbcmModel.predict(point.features)
      (point.label, prediction)
    }

    // precision/recall
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("LR: Precision: " + metrics.precision + " Recall: " + metrics.recall)

    //extract probabilities
//    gbcmModel.clearThreshold()

    //make submission
    val submission = testData.map { point =>
      val prediction = gbcmModel.predict(point.features)
      (point.label, prediction)
    }

    //write submission file
    val collect: Array[(Double, Double)] = submission.collect()
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/submissions/2016_07_03_spark_GBC.csv"))
    pw.write("t_id,probability\n")
    collect.foreach(tuple => pw.write(tuple._1.toInt + "," + tuple._2 + "\n"))
    pw.close()
  }
}