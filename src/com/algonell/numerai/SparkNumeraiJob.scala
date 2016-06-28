package com.algonell.numerai

import java.io.{PrintWriter, File}

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 6/28/16.
  */
object SparkNumeraiJob {
  def main(args: Array[String]) {
    //configure Spark
    val conf = new SparkConf().setAppName("Numerai Spark").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //load train data
    val trainDataFile = sc.textFile("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_training_data.txt")
    val trainData = trainDataFile.map { line =>
      val parts = line.split(',')
      val target: Double = parts(parts.length - 1).toDouble
      val features: Vector = Vectors.dense(parts.slice(0, parts.length - 2).map(_.toDouble))
      LabeledPoint(target, features)
    }.cache()

    //load test data
    val testDataFile = sc.textFile("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/numerai_tournament_data.txt")
    val testData = testDataFile.map { line =>
      val parts = line.split(',')
      val id: Double = parts(0).toDouble
      val features: Vector = Vectors.dense(parts.slice(1, parts.length - 1).map(_.toDouble))
      LabeledPoint(id, features)
    }.cache()

    // split data
    val split = 0.9
    val splits = trainData.randomSplit(Array(split, 1 - split), seed = 11L)
    val train = splits(0).cache()
    val test = splits(1)

    // train model
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
    println("Precision: " + metrics.precision + " Recall: " + metrics.recall)

    //extract probabilities
    lrModel.clearThreshold()

    //make submission
    val submission = testData.map { point =>
      val prediction = lrModel.predict(point.features)
      (point.label, prediction)
    }

    //write submission file
    val collect: Array[(Double, Double)] = submission.collect()
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/2016_06_28_spark_submission.csv"))
    pw.write("t_id,probability\n")
    collect.foreach(tuple => pw.write(tuple._1.toInt + "," + tuple._2 + "\n"))
    pw.close()
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
}

/*
// GradientBoostedTrees model
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.setNumIterations(3) // Note: Use more iterations in practice.
//oostingStrategy.treeStrategy.numClasses = 2
//boostingStrategy.treeStrategy.maxDepth = 5
// Empty categoricalFeaturesInfo indicates all features are continuous.
//boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

val gbcmModel = GradientBoostedTrees.train(trainData, boostingStrategy)

// Evaluate model on test instances and compute test error
val labelAndPreds = testData.map { point =>
  val prediction = gbcmModel.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
println("Test Error = " + testErr)
println("Learned classification GBT model:\n" + gbcmModel.toDebugString)
*/