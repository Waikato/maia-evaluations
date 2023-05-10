package maia.ml.evaluation

import kotlinx.coroutines.runBlocking
import maia.ml.dataset.DataBatch
import maia.ml.dataset.DataStream
import maia.ml.dataset.util.mustHaveEquivalentColumnStructureTo
import maia.ml.dataset.view.readOnlyViewColumns
import maia.ml.evaluation.error.WrongEvaluationTypeException
import maia.ml.learner.Learner
import maia.ml.learner.util.predictInputHeaderColumns

/**
 * Evaluates a stream-learner.
 *
 * @receiver
 *          The learner to evaluate.
 * @param evaluation
 *          The evaluation to perform.
 * @param evaluator
 *          The metric/s to measure during the evaluation.
 *
 * @param L
 *          The type of learner being evaluated.
 * @param M
 *          The type of metric/s being measured.
 *
 * @return
 *          The final metric/s after the evaluation.
 */
fun <L: Learner<DataStream<*>>, M> Array<L>.evaluate(
    evaluation: Evaluation<DataStream<*>>,
    evaluator: StreamEvaluator<L, M>
): Metric<M> = evaluate(
    evaluation,
    evaluator
) { learner, learnerIndex, dataset ->
    // See if there's a harness
    val callback = evaluator.perRowCallback(learner, learnerIndex, dataset)

    // Attach a callback if required
    if (callback != null)
        EvaluationHarness(dataset, callback)
    else
        dataset
}

/**
 * Evaluates a batch-learner.
 *
 * @receiver
 *          The learner to evaluate.
 * @param evaluation
 *          The evaluation to perform.
 * @param evaluator
 *          The metric/s to measure during the evaluation.
 *
 * @param L
 *          The type of learner being evaluated.
 * @param M
 *          The type of metric/s being measured.
 *
 * @return
 *          The final metric/s after the evaluation.
 */
fun <L: Learner<DataBatch<*>>, M> Array<L>.evaluate(
    evaluation: Evaluation<DataBatch<*>>,
    evaluator: BatchEvaluator<L, M>
): Metric<M> = evaluate(
    evaluation,
    evaluator
) { _, _, dataset -> dataset }

/**
 * Common implementation of evaluating streaming and batch operations.
 *
 * @receiver
 *          The learners to evaluate.
 * @param evaluation
 *          The type of evaluation to perform.
 * @param evaluator
 *          The metric/s to measure during the evaluation.
 * @param harnessDataset
 *          Function to place a harness around any training datasets.
 *
 * @param D
 *          The type of dataset the learner requires (batch/stream).
 * @param L
 *          The type of learner being evaluated.
 * @param M
 *          The type of metric/s being measured.
 *
 * @return
 *          The final value of the evaluator's metric for each learner. If the
 *          evaluator is an aggregate evaluator, will only contain one value
 *          for all learners.
 */
private inline fun <D: DataStream<*>, L: Learner<D>, M> Array<L>.evaluate(
    evaluation: Evaluation<D>,
    evaluator: Evaluator<D, L, M>,
    harnessDataset: (L, Int, D) -> D
): Metric<M> {
    // Inform the evaluator of the start of this evaluation
    evaluator.preEvaluate(*this)

    // Use the schedule of evaluation steps as prescribed by the evaluation
    for (step in evaluation.steps(size)) {
        // Get the learner being acted upon in this step
        val learnerIndex = step.learnerIndex
        val learner = this[learnerIndex]

        // Discriminate on step type
        when (step) {
            // Initialisation step
            is Evaluation.Step.Initialise -> {
                // Get the initialisation headers
                val headers = step.headers

                // Inform the evaluator of the incoming initialisation
                evaluator.preInitialise(learner, learnerIndex, headers)

                // Perform the initialisation
                learner.initialise(headers)

                // Make sure the learner initialised to a suitable type for
                // the evaluator
                if (learner.initialisedType isNotSubTypeOf evaluator.type)
                    throw WrongEvaluationTypeException(evaluator, learner.initialisedType)

                // Inform the evaluator that initialisation has completed
                evaluator.postInitialise(learner, learnerIndex, headers)
            }

            // Train step
            is Evaluation.Step.Train -> {
                // Attach the harness to the training dataset
                val dataset = harnessDataset(learner, learnerIndex, step.trainingDataset)

                // Inform the evaluator of the pending train
                evaluator.preTrain(learner, learnerIndex, dataset)

                // Perform the training
                runBlocking { learner.train(dataset) }

                // Inform the evaluator that training has completed
                evaluator.postTrain(learner, learnerIndex, dataset)
            }

            // Predict step
            is Evaluation.Step.Predict -> {
                // Grab the row to predict against
                val row = step.row

                // Make sure the row includes the class information
                row mustHaveEquivalentColumnStructureTo learner.trainHeaders

                // Only feed the non-class information to the learner for prediction
                val predictRow = row.readOnlyViewColumns(learner.predictInputHeaderColumns)

                // Inform the evaluator of the pending prediction
                evaluator.prePredict(learner, learnerIndex, row)

                // Perform the prediction
                val prediction = learner.predict(predictRow)

                // Inform the evaluator that predicting has completed
                evaluator.postPredict(learner, learnerIndex, row, prediction)
            }
        }
    }

    // Inform the evaluator of the end of this evaluation
    evaluator.postEvaluate(*this)

    // Return the evaluator's metric
    return evaluator.metric
}
