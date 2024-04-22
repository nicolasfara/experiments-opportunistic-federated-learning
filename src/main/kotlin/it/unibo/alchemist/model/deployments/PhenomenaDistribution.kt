package it.unibo.alchemist.model.deployments

import it.unibo.alchemist.model.*
import it.unibo.alchemist.model.positions.Euclidean2DPosition

import java.util.stream.Stream
import kotlin.streams.asStream

class PhenomenaDistribution(
    private val xStart: Double,
    private val yStart: Double,
    private val xEnd: Double,
    private val yEnd: Double,
    private val xPhenomenaCount: Int,
    private val yPhenomenaCount: Int,
): Deployment<Position<*>> {
    override fun stream(): Stream<Position<*>> {
        require(xPhenomenaCount > 0)
        require(yPhenomenaCount > 0)
        val xStep = ((xEnd - 1) - xStart) / (xPhenomenaCount + 1)
        val yStep = ((yEnd - 1) - yStart) / (yPhenomenaCount + 1)
        val xPhenomena = (0 until xPhenomenaCount).map { i -> xStep + i * xStep }
        val yPhenomena = (0 until yPhenomenaCount).map { i -> yStep + i * yStep }
        return xPhenomena.flatMap { x -> yPhenomena.map { y -> Euclidean2DPosition(x, y) } }.asSequence().asStream()
    }
}
