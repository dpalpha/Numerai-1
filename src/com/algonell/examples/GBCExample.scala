/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package com.algonell.examples

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.util.MLUtils

object GBCExample {
  def main(args: Array[String]): Unit = {
    //configure Spark
    val conf = new SparkConf().setAppName("GradientBoostedTreesClassificationExample").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //load data
    //val rawData = sc.textFile("dac_sample.txt")
    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt")

    /*
    //transform to LabeledPoint
    val data : RDD[LabeledPoint] = rawData.map {
      line => {
        val parts = line.split("\t")
        val target: Double = parts(0).toDouble
        val features: Vector = Vectors.dense(parts.slice(1, parts.length - 1).map(_.toDouble))
        LabeledPoint(target, features)
      }
    }
    */

    // split
    val SPLIT = 0.8
    val splits = data.randomSplit(Array(SPLIT, 1 - SPLIT), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    // Train a GradientBoostedTrees model.
    // The defaultParams for Classification use LogLoss by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(1000) // Note: Use more iterations in practice.
    boostingStrategy.treeStrategy.setNumClasses(2)
    boostingStrategy.treeStrategy.setMaxDepth(10)
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(Map[Int, Int]())

    val model = GradientBoostedTrees.train(training, boostingStrategy)

    // Evaluate model on test instances and compute test error
    val predictionAndLabels = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / test.count()
    println("Test Error = " + testErr)
    println("Learned classification GBT model:\n" + model.toDebugString)

    // precision/recall
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
    val recall = metrics.recall
    println("Precision = " + precision + " Recall = " + recall)
  }
}

