package it.unibo.alchemist

import it.unibo.alchemist.boundary.swingui.effect.api.DrawLayers
import it.unibo.alchemist.boundary.swingui.effect.impl.AbstractDrawLayers
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.layers.PhenomenaDistribution
import it.unibo.alchemist.model.{Environment, Layer, Position2D}

import java.awt.{Color, Graphics2D}
import java.util

class PhenomenaDistributionEffect extends AbstractDrawLayers {
  private val size = 10
  override def drawLayers[T, P <: Position2D[P]](
      toDraw: util.Collection[Layer[T, P]],
      environment: Environment[T, P],
      graphics: Graphics2D,
      wormhole: Wormhole2D[P]
  ): Unit = {
    val phenomenaDistribution = toDraw
      .stream()
      .filter(_.isInstanceOf[PhenomenaDistribution[P]])
      .map(_.asInstanceOf[PhenomenaDistribution[P]])
      .findFirst()
      .orElseThrow(() =>
        new IllegalArgumentException("No phenomena distribution layer found")
      )

    val positions = phenomenaDistribution.dataByPositions.toList
    positions.foreach { dataset =>
      val guiPosition = wormhole.getViewPoint(dataset._1)
      val color = Color.HSBtoRGB(
        dataset._2.areaId.toFloat / 10,
        1.0f,
        1.0f
      )
      graphics.setColor(new Color(color))
      graphics.fillRect(
        guiPosition.getX.toInt,
        guiPosition.getY.toInt,
        size,
        size
      )
      graphics.setColor(Color.BLACK)
      graphics.drawRect(
        guiPosition.getX.toInt,
        guiPosition.getY.toInt,
        size,
        size
      )
    }
  }

  override def getColorSummary: Color = Color.BLUE
}
