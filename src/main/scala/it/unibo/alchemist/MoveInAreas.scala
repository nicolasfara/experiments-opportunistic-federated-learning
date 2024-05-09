package it.unibo.alchemist

import it.unibo.Utils.{EnvironmentOps, RichNode}
import it.unibo.alchemist.model.{Action, Environment, Node, Reaction}
import it.unibo.alchemist.model.actions.AbstractMoveNode
import it.unibo.alchemist.model.positions.Euclidean2DPosition
import it.unibo.scafi.Sensors

class MoveInAreas[T](
    environment: Environment[T, Euclidean2DPosition],
    node: Node[T],
    speed: Double,
    waitFor: Integer,
    areaOrder: java.util.List[Integer]
) extends AbstractMoveNode[T, Euclidean2DPosition](environment, node, false) {

  private lazy val centersOfAreas = environment.getNodesAsScala
    .map(_.manager)
    .map(node => environment.getPosition(node.node) -> node.get[Integer](Sensors.areaId))
    .groupBy(_._2)
    .map { case (areaId, positions) =>
      areaId -> positions
        .map(_._1)
        .reduce((acc, e) => new Euclidean2DPosition(acc.getX + e.getX, acc.getY + e.getY))
        .div(positions.size)
    }

  var currentWait = waitFor
  override def getNextPosition: Euclidean2DPosition = {
    if (areaOrder.isEmpty) environment.makePosition(0, 0)
    else {
      val nextArea = areaOrder.get(0)
      val target = centersOfAreas(nextArea.toInt)
      val currentPosition = environment.getPosition(node)
      val direction = new Euclidean2DPosition(target.getX - currentPosition.getX, target.getY - currentPosition.getY)
      val distance = currentPosition.distanceTo(target)
      if (distance < speed) {
        currentWait -= 1
        if (currentWait == 0) {
          areaOrder.remove(0)
          currentWait = waitFor
        }
        environment.makePosition(0, 0)
      } else direction.div(distance).div(1 / speed)
    }
  }

  override def cloneAction(node: Node[T], reaction: Reaction[T]): Action[T] =
    new MoveInAreas(environment, node, speed, waitFor, areaOrder)
}
