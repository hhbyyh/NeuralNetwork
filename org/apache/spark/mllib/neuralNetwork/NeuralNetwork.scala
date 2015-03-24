
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

package org.apache.spark.mllib.neuralNetwork

import org.apache.spark.Logging
import org.apache.spark.graphx.{Edge, Graph, _}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import scala.collection.mutable.ArrayBuffer

class ANN private (private val trainingRDD: RDD[(Vector, Vector)],
                   private val hiddenLayersTopology: Array[Int],
                   private var maxIterations: Int) extends Serializable with Logging{

  private val layersCount = hiddenLayersTopology.length + 2 //input and output layers included
  private val MaxLayerCount = 1e10.toLong

  private def getVertexId(layer: Int, index:Int): Long ={
    return layer * MaxLayerCount + index
  }

  private var graph = {
    val topology = 0 +: convertTopology(trainingRDD, hiddenLayersTopology)

    val vertices =
      (1 to layersCount - 1).flatMap( layerIndex =>{
        val layerLength = topology(layerIndex)
        (0 to layerLength).map( i => {
          val vertexId = getVertexId(layerIndex, i)
          (vertexId, (layerIndex, i, 1.0, 0.0)) //layer, index, value, delta
        })
      })
    .union((1 to topology(layersCount)).map( i => {
        val vertexId = getVertexId(layersCount, i)
        (vertexId, (layersCount, i, 1.0, 0.0))
      })) //last layer without bias

    val edges = (2 to layersCount).flatMap(layerIndex =>{
      val preLayer = layerIndex - 1
      val prelayerLength = topology(layerIndex - 1)
      val layerLength = topology(layerIndex)

      val buffer = new ArrayBuffer[Edge[Double]]()
      for(target <- 1 to layerLength)
        for(src <- 0 to prelayerLength){
          val srcId = getVertexId(preLayer, src)
          val targetId = getVertexId(layerIndex, target)
          buffer += Edge(srcId, targetId, scala.util.Random.nextDouble())
        }
      buffer
    })

    val verticesRdd: RDD[(VertexId, (Int, Int, Double, Double))] = trainingRDD.context.parallelize(vertices)
    val edgesRdd: RDD[Edge[Double]] = trainingRDD.context.parallelize(edges)

    Graph(verticesRdd, edgesRdd).partitionBy(PartitionStrategy.CanonicalRandomVertexCut)
  }

  var forwardCount = graph.vertices.sparkContext.accumulator(0)

  private def convertTopology(input: RDD[(Vector,Vector)],
                              hiddenLayersTopology: Array[Int] ): Array[Int] = {
    val firstElt = input.first
    firstElt._1.size +: hiddenLayersTopology :+ firstElt._2.size
  }

  def run(trainingRDD: RDD[(Vector, Vector)]): Unit ={
    var i = 1
    val data = trainingRDD.collect()
    while(i < maxIterations){
      var diff = 0.0
      data.foreach(sample => {
        val d = this.Epoch(sample._1, sample._2)
        diff += d
        println(d)
      })
      logError(s"iteration $i get " + diff)
      i += 1
    }
    println("forwardCount: " + forwardCount.value)
  }


  def Epoch(input: Vector, output: Vector): Double = {
    val preGraph = graph
    graph.cache()
    assignInput(input)
    for(i <- 1 to (layersCount - 1)){
      forward(i)
    }
    val diff = ComputeDeltaForLastLayer(output)

    for(i <- layersCount until 2 by -1){
      backporgation(i)
    }
    updateWeights()
    preGraph.unpersist()
    diff
  }

  private def assignInput(input: Vector): Unit = {

    val in = graph.vertices.context.parallelize(input.toArray.zipWithIndex)
    val inputRDD = in.map( x => (x._2 + 1, x._1)).map(x =>{
      val id = getVertexId(1, x._1)
      (id, x._2)
    })

    graph = graph.joinVertices(inputRDD){
      (id, oldVal, input) => (oldVal._1, oldVal._2, input, oldVal._4)
    }
  }

  /**
   * Feed forward from layerIndex to layerIndex + 1
   */
  private def forward(layerIndex: Int):Unit ={

    val sumRdd:VertexRDD[Double] = graph.subgraph(edge => edge.srcAttr._1 == layerIndex).aggregateMessages[Double](
      triplet => {
        val value = triplet.srcAttr._3
        val weight = triplet.attr
        triplet.sendToDst(value * weight)
      }, _ + _, TripletFields.Src
    )
    forwardCount += 1

    graph = graph.joinVertices(sumRdd){
      (id, oldRank, msgSum) => (oldRank._1, oldRank._2, breeze.numerics.sigmoid(msgSum), oldRank._4)
    }

  }

  /**
   * from layerIndex to layerIndex - 1
   */
  private def backporgation(layerIndex: Int): Unit ={
    val deltaRdd = graph.subgraph(edge => edge.dstAttr._1 == layerIndex).aggregateMessages[Double](
      triplet => {
        val delta = triplet.dstAttr._4
        val weight = triplet.attr
        triplet.sendToSrc(delta * weight)
      }, _ + _, TripletFields.Dst
    )

    graph = graph.joinVertices(deltaRdd){
      (id, oldValue, deltaSum) => {
        val e = deltaSum
        (oldValue._1, oldValue._2, oldValue._3, e)     // update delta
      }
    }
  }

  private def updateWeights(): Unit ={
    val eta = 10

    graph = graph.mapTriplets(triplet =>{
      val delta = triplet.dstAttr._4
      val y = triplet.dstAttr._3
      val x = triplet.srcAttr._3
      val newWeight = triplet.attr + eta * delta * y * (1.0 - y) * x
      newWeight
    })
  }

  private def ComputeDeltaForLastLayer(output: Vector): Double ={
    var sampleDelta = graph.vertices.sparkContext.accumulator(0.0)
    graph = graph.mapVertices( (id, attr) => {
      if(attr._1 != layersCount){
        attr
      }
      else{
        val index = attr._2
        val d = output(index - 1)
        val y = attr._3
        val delta = (d - y)
        sampleDelta += (d - y) * (d - y)
        (attr._1, attr._2, attr._3, delta)
      }
    })
    graph.vertices.count()
    sampleDelta.value * 0.5
  }

  def predict(input: Vector): Vector ={
    assignInput(input)
    for(i <- 1 to (layersCount - 1)){
      forward(i)
    }
    val result = graph.vertices.filter( x=> x._2._1 == layersCount).map(x => x._2).map(x => (x._2, x._3))
    val doubles = result.sortBy(x => x._1).map(x => x._2).collect()
    Vectors.dense(doubles)
  }
}


/**
 * Top level methods for training the artificial neural network (ANN)
 */
object ANN {

  private val defaultTolerance: Double = 1e-4

  def train(trainingRDD: RDD[(Vector, Vector)],
            hiddenLayersTopology: Array[Int],
            maxNumIterations: Int) : ANN = {

    val ann = new ANN(trainingRDD, hiddenLayersTopology, maxNumIterations)
    ann.run(trainingRDD)
    return ann
  }

}
