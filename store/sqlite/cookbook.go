package sqlite

import (
	_ "embed"
	"regexp"
	"strings"
)

//go:embed cookbook.sql
var cookbookSQL string

var cookbookHeader = regexp.MustCompile(`(?m)^-- (Q\d+):.*$`)

// cookbookQuery returns the statement of the cookbook query named name ("Q6").
func cookbookQuery(name string) string {
	locs := cookbookHeader.FindAllStringSubmatchIndex(cookbookSQL, -1)
	for i, loc := range locs {
		if cookbookSQL[loc[2]:loc[3]] != name {
			continue
		}
		end := len(cookbookSQL)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		return strings.TrimSpace(cookbookSQL[loc[0]:end])
	}
	panic("sqlite: no cookbook query " + name)
}
