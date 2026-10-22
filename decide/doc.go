// Package decide is the vocabulary for decisions: given a state and a set
// of typed questions, a [Decider] returns a [Response] of per-question
// answers. The package holds no HTTP code and names no backend. Package
// decide/systemone is the System One wire backend.
//
// A decision model is not a chat model. It has no messages, tools, or
// streaming, so a [Decider] is not an [llmkit.Client].
//
// # Questions
//
// [Question] is a sealed interface with three implementations:
// [Noul] (binary belief in [0, 1]), [Choice] (pick one option from a
// map of descriptions), and [Score] (rate against an ordered legend of
// at least two levels). [Questions] indexes one Ask's questions by id.
//
// State, instructions, and descriptions accept Go strings and anything
// that marshals to a JSON object or array. Descriptions and criteria
// also accept null. JSON numbers and booleans are refused.
//
// [Normalize] validates a state and its questions and returns them in the
// root vocabulary ([llmkit.DecisionQuestion]), sorted by id, as raw JSON.
// A backend renders its own request from that output. A refusal is an
// [*llmkit.APIError] with Kind [llmkit.ErrInvalidRequest], StatusCode 0,
// Provider "decide", and a "<path>: <problem>" message; the backend
// re-tags Provider to its own name.
//
// # Answers
//
// [Response] splits answers by question kind. [Response.Nouls] maps a
// noul id to its 0..1 belief. [Response.Choices] maps a choice id to a
// [ChoiceAnswer]. [Response.Scores] maps a score id to a [ScoreAnswer]
// whose Levels and Probabilities are index-aligned slices. Maps stay nil
// for kinds the Ask did not include.
//
// Read answers through the accessors [Response.Noul], [Response.Choice],
// and [Response.Score]. Each returns an error wrapping [ErrNotAsked] when
// the id was not asked as that kind, so a typo is an error, not a zero.
//
// A backend that reports no distribution leaves Probabilities nil and
// Confidence 0. A backend never fabricates either. [Check] reports whether
// a Response answers a [Questions] set exactly.
//
// # Deciders
//
// [Decider] is the interface a backend implements. Its errors wrap a root
// llmkit sentinel (llmkit.Err*) or chain the context's error. A successful
// Response satisfies [Check] against the questions asked.
package decide
