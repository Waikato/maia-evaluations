package maia.ml.evaluation

import maia.util.formatPretty

/**
 * TODO: What class does.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
sealed class Metric<out M> {



    class PerLearner<out M>(
        private vararg val values: M
    ): Metric<M>() {
        operator fun get(index: Int): M = values[index]
        val size: Int = values.size
        operator fun iterator(): Iterator<M> = values.iterator()
        override fun toString() : String {
            return values.indices.joinToString { index: Int ->
                "$index: ${values[index]}"
            }
        }
    }

    class Aggregate<out M>(
        val value: M
    ): Metric<M>() {
        override fun toString() : String {
            return value.toString()
        }
    }

    class Group<out M> internal constructor(
        private val map: Map<String, Metric<M>>
    ): Metric<M>(), Iterable<String> {
        val size: Int = map.size
        operator fun get(name: String) = map[name]
        override fun iterator() : Iterator<String> = map.keys.iterator()
        override fun toString() : String {
            return map.formatPretty()
        }
    }
}
