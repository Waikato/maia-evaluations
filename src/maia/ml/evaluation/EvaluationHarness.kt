package maia.ml.evaluation

import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream

typealias EvaluationHarnessCallback = (row: DataRow) -> Unit

/**
 * Harness for attaching callbacks to each sourced data-row when using a
 * streaming dataset.
 *
 * @param stream
 *          The streaming dataset.
 * @param callback
 *          The callback to call for each sourced row.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class EvaluationHarness(
    private val stream: DataStream<*>,
    private val callback: EvaluationHarnessCallback
): DataStream<DataRow> by stream {

    /** Storage for a single instance of the harnessed iterator. */
    private lateinit var harness: HarnessIterator

    override fun rowIterator() : Iterator<DataRow> {
        if (!this::harness.isInitialized)
            harness = HarnessIterator(stream.rowIterator())
        return harness
    }

    /**
     * Harness over the source's row-iterator.
     *
     * @param source
     *          The row-iterator of the source data-stream.
     */
    private inner class HarnessIterator(
        private val source: Iterator<DataRow>
    ): Iterator<DataRow> by source {

        override fun next() : DataRow {
            // Get the next row from the source
            val next = source.next()

            // Inform the evaluator that a new row has been produced
            callback(next)

            return next
        }

    }

}
