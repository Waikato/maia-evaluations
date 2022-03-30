package maia.ml.evaluation.standard.evaluation

import maia.ml.dataset.DataRow
import maia.ml.dataset.DataStream
import maia.ml.dataset.view.viewAsDataBatch
import maia.ml.evaluation.Evaluation
import maia.util.ensureHasNext

/**
 * Performs an evaluate-prequential evaluation. The learner performs a
 * prediction on each data-row before training on it.
 *
 * @param stream
 *          The stream to perform the evaluation on.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class EvaluatePrequential(
    private val stream: DataStream<*>
) : Evaluation<DataStream<*>> {

    override fun steps(numLearners : Int) : Iterator<Evaluation.Step<DataStream<*>>> {

        return iterator {
            // Initialise all learners
            for (learnerIndex in 0 until numLearners)
                yield(Evaluation.Step.Initialise(learnerIndex, stream.headers))

            // Predict then train each learner on each row
            stream.rowIterator().forEach { row ->
                for (learnerIndex in 0 until numLearners) {
                    // Predict
                    yield(Evaluation.Step.Predict(learnerIndex, row))

                    // Train
                    yield(Evaluation.Step.Train(learnerIndex, row.viewAsDataBatch()))
                }
            }
        }
    }
}
