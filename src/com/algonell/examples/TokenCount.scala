package com.algonell.examples

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by andrewkreimer on 6/14/16.
  */
object TokenCount {
  def main(args: Array[String]) {
    //configure Spark
    val sc = new SparkContext(new SparkConf().setAppName("Simple Application").setMaster("local[*]"))

    //load data
//    val files: RDD[String] = sc.textFile("dac_sample.txt")
    val files: List[RDD[String]] = List(sc.textFile("dac_sample.txt"))

    //simple one stage job
    for (file <- files) {
      val count = file.
        map(line => line + "?").
        flatMap(line => line.split("\t")).
        filter(line => line.contains("?")).
        count()
      println(count)
    }

    //multiple stage job
    for (file <- files){
      //complex job
      val lines = file.map(_.split("\n"))
      println("lines: " + lines.count())

      val tokenized = file.flatMap(_.split('\t'))
      println("tokens: " + tokenized.count())

      val wordCounts = tokenized.map((_, 1)).reduceByKey(_ + _)
      val filtered = wordCounts.filter(_._2 >= 10000)
      println("word counts:")
      filtered.foreach(println)
    }
  }
}
