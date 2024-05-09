package it.unibo.alchemist

import it.unibo.alchemist.boundary.swingui.effect.impl.AbstractDrawLayers
import it.unibo.alchemist.boundary.ui.api.Wormhole2D
import it.unibo.alchemist.model.layers.IdPhenomenaLayer
import it.unibo.alchemist.model.{Environment, Layer, Position2D}

import java.awt.{Color, Graphics2D, Point}
import java.lang.Math.ceil
import java.util

class SampleGeneralLayer extends AbstractDrawLayers {

  val serialVersionUID: Long = 1L
  private val MinSamples = 10
  private val MaxSamples = 400

  override def drawLayers[T, P <: Position2D[P]](
      toDraw: util.Collection[Layer[T, P]],
      environment: Environment[T, P],
      graphics: Graphics2D,
      wormhole: Wormhole2D[P]
  ): Unit = {
    val viewSize = wormhole.getViewSize
    val (viewStartX, viewStartY) = (0, 0)
    val (viewEndX, viewEndY) = (ceil(viewSize.getWidth).toInt, ceil(viewSize.getHeight).toInt)

    val stepX = (viewEndX - viewStartX) / 100
    val stepY = (viewEndY - viewStartY) / 100
    val phenomenaDistributionLayer = toDraw
      .stream()
      .filter(_.isInstanceOf[IdPhenomenaLayer[P]])
      .map(_.asInstanceOf[IdPhenomenaLayer[P]])
      .findFirst()
      .orElseThrow(() => new IllegalArgumentException("No phenomena distribution layer found"))
    for {
      i1 <- viewStartX until viewEndX by stepX
      j1 <- viewStartY until viewEndY by stepY
    } {
      val (i2, j2) = (i1 + stepX, j1 + stepY)
      val points = List(new Point(i1, j1), new Point(i1, j2), new Point(i2, j1), new Point(i2, j2))
      val values = points.map(p => phenomenaDistributionLayer.getValue(wormhole.getEnvPoint(p)))
      val average = values.max
      val color = Color.getHSBColor(average.toFloat, 0.5f, 0.9f)
      val alphed = new Color(color.getRed, color.getGreen, color.getBlue, 128)
      graphics.setColor(alphed)
      graphics.fillRect(i1, j1, i2 - i1, j2 - j1)
    }
  }

}
