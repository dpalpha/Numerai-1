package com.algonell.examples

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 1/28/16.
  */
object LRExample2 {
  def main(args: Array[String]) {
    //configure Spark
    val conf = new SparkConf().setAppName("Logistic Regression").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //load data
    //val rawData = sc.textFile("dac_sample.txt")
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    // split
    val SPLIT = 0.8
    val splits = data.randomSplit(Array(SPLIT, 1 - SPLIT), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    // train model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)

    // get predictions on test set
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // precision/recall
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    val recall = metrics.recall
    println("Precision = " + precision + " Recall = " + recall)

    //show PMML
    println(model.toPMML())
    model.toPMML("LR.xml")
  }
}
