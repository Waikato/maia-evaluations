package maia.ml.evaluation

import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.WithColumns

/**
 * Interface for classes which describe the steps of an evaluation of
 * learners.
 *
 * @param D
 *          The type of dataset this evaluation can be used with.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
interface Evaluation<out D: DataStream<*>> {

    /** Gets an iterator over the steps to perform for the evaluation. */
    fun steps(numLearners: Int): Iterator<Step<D>>

    /**
     * Base-class representing an individual step in the process of an
     * evaluation.
     *
     * @param learnerIndex
     *          The index of the learner to apply the step to.
     *
     * @param D
     *          The type of dataset this step can be used with.
     */
    sealed class Step<out D: DataStream<*>>(val learnerIndex : Int) {

        /**
         * Evaluation step to initialise the learner on a particular column-
         * structure.
         *
         * @param learnerIndex
         *          The index of the learner to apply the step to.
         * @param headers
         *          The column structure to initialise the learner with.
         */
        class Initialise(
            learnerIndex: Int,
            val headers: WithColumns
        ) : Step<DataStream<*>>(learnerIndex)

        /**
         * Evaluation step to train the learner on a particular dataset.
         *
         * @param learnerIndex
         *          The index of the learner to apply the step to.
         * @param trainingDataset
         *          The dataset to train the learner on.
         */
        class Train<out D: DataStream<*>>(
            learnerIndex: Int,
            val trainingDataset: D
        ) : Step<D>(learnerIndex)

        /**
         * Evaluation step to use the learner to perform a prediction on a
         * particular row of data.
         *
         * @param learnerIndex
         *          The index of the learner to apply the step to.
         * @param row
         *          The row of data to perform the prediction on.
         */
        class Predict(
            learnerIndex: Int,
            val row: DataRow
        ) : Step<DataStream<*>>(learnerIndex)

    }

}
