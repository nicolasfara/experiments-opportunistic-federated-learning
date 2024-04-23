package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.scafi.interop.PythonModules.utils

class ModelInitializationReaction  [T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int
) extends AbstractGlobalReaction(environment, distribution){

  override protected def executeBeforeUpdateDistribution(): Unit = {
    utils.init_cnn(seed.asInstanceOf[Double].toInt)
    val model = utils.cnn_loader(seed)
    nodes.foreach(n => n.setConcentration(new SimpleMolecule("Model"), model.state_dict().asInstanceOf[T]))
  }

}
