package com.algonell.numerai

import java.io.{PrintWriter, File}
import java.text.SimpleDateFormat
import java.util.{Date, Calendar}

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 6/28/16.
  */
object SparkNumeraiRddJob {
  val TODAY = getToday

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

    trainLRModel(testData, train, test)
    //trainGBCModel(testData, train, test)

    println(TODAY)
  }

  def trainLRModel(testData: RDD[LabeledPoint], train: RDD[LabeledPoint], test: RDD[LabeledPoint]) = {
    // train LR model
    val lrAlgorithm = new LogisticRegressionWithLBFGS()
    lrAlgorithm.setNumClasses(2)
    lrAlgorithm.optimizer.setNumIterations(1000)
    lrAlgorithm.optimizer.setRegParam(0.0001)
    lrAlgorithm.optimizer.setConvergenceTol(0.0001)
    lrAlgorithm.optimizer.setNumCorrections(1000)
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
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/submissions/" + TODAY + "_spark_LR.csv"))
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
    val pw = new PrintWriter(new File("/Users/andrewkreimer/Documents/ML/Kaggle/Numerai/submissions/" + TODAY + "_spark_GBC.csv"))
    pw.write("t_id,probability\n")
    collect.foreach(tuple => pw.write(tuple._1.toInt + "," + tuple._2 + "\n"))
    pw.close()
  }

  def getToday: String = {
    val dateFormatter = new SimpleDateFormat("yyy_MM_hh")
    dateFormatter.format(new Date())
  }
}