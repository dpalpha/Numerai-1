package com.algonell.examples

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 1/27/16.
  */
object CriteoSample {
  def main(args: Array[String]) {
    //configure Spark
    val sc = new SparkContext(new SparkConf().setAppName("Simple Application").setMaster("local[*]"))

    //load data
    var criteoData = sc.textFile("dac_sample.txt").cache()

    //show original
    criteoData.take(5).foreach(println)

    //convert to csv lines
    criteoData = criteoData.map(x => x.split("\t").mkString(","))
    criteoData.take(5).foreach(println)

    val tokenized = criteoData.flatMap(_.split(','))
    val wordCounts = tokenized.map((_, 1)).reduceByKey(_ + _)
    val filtered = wordCounts.filter(_._2 >= 1000)
    val charCounts = filtered.flatMap(_._1.toCharArray).map((_, 1)).reduceByKey(_ + _)
    charCounts.collect().take(5).foreach(println)
  }
}
