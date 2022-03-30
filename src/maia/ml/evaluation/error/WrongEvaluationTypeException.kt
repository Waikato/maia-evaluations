package maia.ml.evaluation.error

import maia.ml.evaluation.Evaluator
import maia.ml.learner.type.LearnerType

/**
 * Exception for when an evaluator can't evaluate a given learner type.
 *
 * @param evaluator
 *          The evaluator that has the error.
 * @param actualType
 *          The type of learner that the evaluator couldn't handle.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class WrongEvaluationTypeException(
    evaluator: Evaluator<*, *, *>,
    actualType: LearnerType
): Exception(
    "Learner is the wrong type for evaluator ${evaluator.name}:" +
            "should be ${evaluator.type} but got $actualType"
)
