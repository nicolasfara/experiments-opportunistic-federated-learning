package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.{Environment, Node, Position}
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote
import scala.jdk.CollectionConverters.IteratorHasAsScala

class TestSetEvaluation[P <: Position[P]](seed: Double, epochs: Int, areas: Int, dataShuffle: Boolean)
    extends OutputMonitor[Any, P] {

  protected val batch_size = 64

  protected def nodes(environment: Environment[Any, P]): List[Node[Any]] =
    environment.getNodes.iterator().asScala.toList

  protected def evaluate(weights: py.Dynamic, data: py.Dynamic): Double = {
    val model = utils.nn_from_weights(weights)
    val result = utils.evaluate(model, data, batch_size, seed)
    val accuracy = py"$result[0]".as[Double]
    accuracy
  }

  def cleanPythonObjects(): Unit = {
    val gc = py.module("gc")
    val pythonObjects = py"list($gc.get_objects())".as[Seq[py.Dynamic]]
    for (elem <- pythonObjects) {
      try {
        py"del $elem"
      } catch {
        case e: Exception => println(e)
      }
    }
    gc.collect()
    Runtime.getRuntime.gc()
  }
}
