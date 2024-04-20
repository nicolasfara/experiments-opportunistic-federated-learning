package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.scafi.interop.PythonModules.utils
import me.shadaj.scalapy.py
import scala.util.Random

class CentralLearnerReaction [T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    seed: Int,
    clientsFraction: Double
) extends AbstractGlobalReaction(environment, distribution){

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val clients = (environment.getNodes.size() * clientsFraction).toInt
    val localModels = getModels(clients)
    val globalModel = averageWeights(localModels)
    nodes.foreach( n => n.setConcentration(new SimpleMolecule("GlobalModel"), globalModel.asInstanceOf[T]) )
  }
  
  private def getModels(requiredModels: Int): List[py.Dynamic] = {
    val models = nodes
      .map(_.getConcentration(new SimpleMolecule("LocalModel")).asInstanceOf[py.Dynamic])
    new Random(seed).shuffle(models).take(requiredModels)
  }

  private def averageWeights(localModels: Seq[py.Dynamic]): py.Dynamic = {
    val averageWeights =
      utils.average_weights(localModels.toPythonProxy)
    val freshNN = utils.cnn_loader(seed)
    freshNN.load_state_dict(averageWeights)
    freshNN
  }

}
