package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model._
import it.unibo.scafi.interop.PythonModules
import it.unibo.Utils._
import it.unibo.scafi.Sensors

class DataDistributionReaction[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int,
    areas: Int,
    dataShuffle: Boolean,
    dataFraction: Double
) extends AbstractGlobalReaction(environment, distribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val nodesCount = environment.getNodes.size()
    val dataDistribution = PythonModules.utils
      .dataset_to_nodes_partitioning(nodesCount, areas, seed, dataShuffle, dataFraction)
      .as[Map[Int, (List[Int], Set[Int])]]

  }

  override def initializationComplete(
      time: Time,
      environment: Environment[T, _]
  ): Unit = getTimeDistribution.update(Time.INFINITY, true, 0.0, environment)
}
