package decide

// Question is the sealed set of decision questions: [Noul], [Choice], and
// [Score]. The unexported method keeps the set closed to code outside this
// package.
type Question interface{ question() }

// Noul asks a binary belief question: how strongly the state supports the
// True description over the False one. Instructions is required (string,
// object, or array). True and False are optional; a nil side omits its key
// and both nil omit the whole criteria object.
type Noul struct {
	Instructions any
	True, False  any
}

// Choice asks which option best fits the state. Instructions is required.
// Options maps option id to description and is required and non-empty; a nil
// value is a JSON null description.
type Choice struct {
	Instructions any
	Options      map[string]any
}

// Score asks for a rating against an ordered legend. Instructions is
// required. Levels is the ordered list of level descriptions, from lowest to
// highest; it is required and must hold at least two.
type Score struct {
	Instructions any
	Levels       []any
}

func (Noul) question()   {}
func (Choice) question() {}
func (Score) question()  {}
