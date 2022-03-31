package maia.ml.evaluation

import maia.util.format
import maia.util.indexIterator
import maia.util.map
import maia.util.times

/**
 * Containers for metrics measured by evaluators.
 *
 * @param M
 *          The type of the metric.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
sealed class Metric<out M> {

    /**
     * Formats the metric to an indented string.
     *
     * @param indent
     *          The level of indentation to use.
     * @return
     *          The formatted string.
     */
    abstract fun format(indent: Int): String

    final override fun toString() : String {
        return format(0)
    }

    /**
     * Gets an indentation string with the specified level of indentation.
     *
     * @param indent
     *          The level of indentation.
     * @return
     *          The indentation string.
     */
    protected fun indentString(indent: Int) = "  " * indent * 2

    /**
     * A metric that is measured per-learner in an evaluation.
     *
     * @param values
     *          The measured values, one per learner.
     *
     * @param M
     *          The type of the metric.
     */
    class PerLearner<out M>(
        private vararg val values: M
    ): Metric<M>(), Iterable<M> {
        val size: Int = values.size
        operator fun get(index: Int): M = values[index]
        override operator fun iterator(): Iterator<M> = values.iterator()
        override fun format(indent: Int) : String {
            val indentString = indentString(indent)
            return values.indices.joinToString(
                separator = "\n$indentString", prefix = indentString
            ) { index: Int ->
                "Learner $index:\n$indentString  ${values[index]}"
            }
        }
    }

    /**
     * A metric that is aggregated over all learners in an evaluation.
     *
     * @param value
     *          The value of the aggregated metric.
     *
     * @param M
     *          The type of the metric.
     */
    class Aggregate<out M>(
        val value: M
    ): Metric<M>() {
        override fun format(indent : Int) : String {
            return "${indentString(indent)}$value"
        }
    }

    /**
     * A group of metrics.
     *
     * @param map
     *          The map from metric names to the metrics themselves.
     *
     * @param M
     *          The type of the metrics.
     */
    class Group<out M> internal constructor(
        private val map: LinkedHashMap<String, Metric<M>>
    ): Metric<M>(), Iterable<String> {
        constructor(map: Map<String, Metric<M>>): this(LinkedHashMap(map))
        val size: Int = map.size
        operator fun get(name: String) = map[name]
        override fun iterator() : Iterator<String> = object : Iterator<String> by map.keys.iterator() {}
        override fun format(indent : Int) : String {
            val indentString = indentString(indent)
            return map.format(
                "$indentString{\n",
                "\n$indentString}",
                ":\n",
                "\n",
                { "$indentString  $it" },
                { it.format(indent + 1) }
            )
        }
    }

    /**
     * A group of metrics per learner in the evaluation.
     *
     * @param maps
     *          The maps from metric names to the metrics themselves, one per learner.
     *
     * @param M
     *          The type of the metric.
     */
    class PerLearnerGroup<out M> internal constructor(
        private vararg val maps: LinkedHashMap<String, M>
    ): Metric<M>(), Iterable<Map<String, M>> {
        constructor(vararg maps: Map<String, M>): this(*maps.map { LinkedHashMap(it) }.toTypedArray())
        val size: Int = maps.size
        operator fun get(index: Int) = object : Map<String, M> by maps[index] {}
        override operator fun iterator() : Iterator<Map<String, M>> = indexIterator(size).map { this[it] }
        override fun format(indent : Int) : String {
            val indentString = indentString(indent)
            return maps.indices.joinToString(
                separator = "\n$indentString", prefix = indentString
            ) { index: Int ->
                "Learner $index:\n$indentString  ${
                    maps[index].entries.joinToString(separator = "\n$indentString  ") { 
                        "${it.key}:\n$indentString    ${it.value}"
                    }
                }"
            }
        }
    }
}
