import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.neuralNetwork.ANN
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by yuhao on 3/16/15.
 */
object TrainOdd{

  def main(args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("Parallel ANN").setMaster("local[4]")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("/home/yuhao/workspace/github/bgreeven/NeuralNetwork/output")

    val arr = new scala.collection.mutable.ArrayBuffer[(Vector, Vector)]()

    arr += new Tuple2(Vectors.dense(0, 1), Vectors.dense(1))
    arr += new Tuple2(Vectors.dense(1, 1), Vectors.dense(1))
    arr += new Tuple2(Vectors.dense(1, 0), Vectors.dense(0))
    arr += new Tuple2(Vectors.dense(0, 0), Vectors.dense(0))

    val startTime = System.nanoTime()
    val ann = ANN.train(sc.parallelize(arr), Array(2), 100)
    val elapsed = (System.nanoTime() - startTime) / 1e9

    println(s"Finished training NN model.  Summary:")
    println(s"\t Training time: $elapsed sec")

    var pre = ann.predict(Vectors.dense(1, 1))
    println(pre)

    pre = ann.predict(Vectors.dense(0, 0))
    println(pre)
    sc.stop()
  }
}
