package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.layers.Dataset
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.alchemist.exporter.TestDataExporter
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.PyQuote

class DistributedTestSetEvaluation[P <: Position[P]](
    seed: Double,
    epochs: Int,
    aggregateLocalEvery: Int,
    areas: Int,
    dataShuffle: Boolean,
    lossThreshold: Double
) extends TestSetEvaluation[P](seed, epochs, areas, dataShuffle) {

  override def finished(
      environment: Environment[Any, P],
      time: Time,
      step: Long
  ): Unit = {
    val layer =
      environment.getLayer(new SimpleMolecule(Sensors.testsetPhenomena)).get()
    val accuracies =
      nodes(environment)
        .map(node => {
          val weights = node
            .getConcentration(new SimpleMolecule(Sensors.model))
            .asInstanceOf[py.Dynamic]
          val data = layer
            .getValue(environment.getPosition(node))
            .asInstanceOf[Dataset]
          (weights, data.trainingData)
        })
        .map { case (w, d) => evaluate(w, d) }
    TestDataExporter.CSVExport(
      accuracies,
      s"data-test/test_accuracy_seed-${seed}_epochs-${epochs}" +
        s"_aggregateLocalEvery-${aggregateLocalEvery}_areas-${areas}" +
        s"_batchSize-${batch_size}_dataShuffle-${dataShuffle}" +
        s"_lossThreshold-${lossThreshold}"
    )
    val gc = py.module("gc")
    try {
      val pythonObjects = py"list($gc.get_objects())".as[Seq[py.Dynamic]]
      for (elem <- pythonObjects) {
        py"del $elem"
      }
      gc.collect()
    } catch {
      case e: Exception => println(e)
    }
  }

}
