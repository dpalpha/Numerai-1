package com.algonell.numerai

import java.io.{File, PrintWriter}

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 6/28/16.
  */
object SparkNumeraiDataFrameJob {
  def main(args: Array[String]) {
    //configure Spark
    val conf = new SparkConf().setAppName("Numerai Spark").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = Array(Row(Vectors.dense(-2.0, 2.3, 0.0)))

    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

    val dataRDD = sc.parallelize(data)
    val dataset = sqlContext.createDataFrame(dataRDD, StructType(Array(attrGroup.toStructField())))

    val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

    slicer.setIndices(Array(1)).setNames(Array("f3"))
    // or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))

    val output = slicer.transform(dataset)
    println(output.select("userFeatures", "features").first())

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
    trainGBCModel(testData, train, test)
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