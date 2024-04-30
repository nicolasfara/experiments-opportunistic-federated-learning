package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.layers.Dataset
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Node, Position, Time}
import it.unibo.scafi.Sensors
import it.unibo.alchemist.exporter.TestDataExporter
import me.shadaj.scalapy.py

class DistributedTestSetEvaluation[P <: Position[P]] extends TestSetEvaluation[P] {

  override def finished(
      environment: Environment[Any, P],
      time: Time,
      step: Long
  ): Unit = {
    println("Starting evaluation...")
    val layer =
      environment.getLayer(new SimpleMolecule(Sensors.testsetPhenomena)).get()
    val accuracies =
      nodes(environment)
        .map(node => {
          println(s"Node ${node.getId} is evaluating...")
          val weights = node
            .getConcentration(new SimpleMolecule(Sensors.model))
            .asInstanceOf[py.Dynamic]
          val data = layer
            .getValue(environment.getPosition(node))
            .asInstanceOf[Dataset]
          (weights, data.trainingData)
        })
        .map { case (w, d) => evaluate(w, d) }
    TestDataExporter.CSVExport(accuracies, "data/test-accuracy")
  }

}
