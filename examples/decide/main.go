// Command decide runs a mixed question set — a Noul, a Choice, and a Score —
// against a support-ticket state and prints the normalized answer kinds as
// indented JSON.
//
// Usage:
//
//	LLMKIT_TYPESAFE_API_KEY=... LLMKIT_TYPESAFE_MODEL=jev-latest \
//	  go run ./examples/decide
//
// LLMKIT_TYPESAFE_BASE_URL optionally targets a gateway; unset means the
// vendor's production endpoint. Without the two required variables the
// program prints usage and exits 1 without touching the network.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/dpoage/llmkit"
	"github.com/dpoage/llmkit/decide"
)

const usage = `missing environment:
  LLMKIT_TYPESAFE_API_KEY   TypeSafe API key (Jev decision model)
  LLMKIT_TYPESAFE_MODEL     model name, e.g. jev-latest
  LLMKIT_TYPESAFE_BASE_URL  optional; gateway base URL, default https://api.typesafe.ai

set the variables above, then re-run:
  go run ./examples/decide`

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "decide:", err)
		os.Exit(1)
	}
}

func run() error {
	key := os.Getenv("LLMKIT_TYPESAFE_API_KEY")
	model := os.Getenv("LLMKIT_TYPESAFE_MODEL")
	if key == "" || model == "" {
		return errors.New(usage)
	}

	client, err := decide.New(decide.Config{
		APIKey:  key,
		Model:   model,
		BaseURL: os.Getenv("LLMKIT_TYPESAFE_BASE_URL"),
	})
	if err != nil {
		return fmt.Errorf("build client: %w", err)
	}

	resp, err := client.Ask(context.Background(),
		"Customer reports a paper jam error on a networked office printer. They removed the "+
			"jammed sheet and restarted the device, but the error returns within minutes and the "+
			"duplex unit sounds like it is grinding.",
		decide.Questions{
			"hardware_fault": decide.Noul{
				Instructions: "True if the evidence points at a physical defect rather than consumables, configuration, or software.",
				True:         "a physical or firmware defect is the likely cause",
				False:        "consumables, configuration, or software explain the symptom",
			},
			"next_action": decide.Choice{
				Instructions: "Which single next step best serves the customer?",
				Options: map[string]any{
					"dispatch_technician": "Send a field technician; likely hardware replacement.",
					"guided_reclean":      "Walk the customer through a duplex-roller cleaning session.",
					"replace_device":      "The unit is under warranty; ship a replacement directly.",
				},
			},
			"severity": decide.Score{
				Instructions: "Rate the operational severity of this ticket.",
				Levels: []any{
					"Cosmetic or minor inconvenience; no user impact.",
					"Annoying but a workaround exists.",
					"Blocking work for the reporting user.",
					"Blocking a team, with data loss or a security angle.",
				},
			},
		})
	if err != nil {
		return fmt.Errorf("ask: %w", err)
	}

	noul, ok := resp.Nouls["hardware_fault"]
	if !ok {
		return errors.New("ask: no answer for hardware_fault")
	}
	choice, ok := resp.Choices["next_action"]
	if !ok {
		return errors.New("ask: no answer for next_action")
	}
	score, ok := resp.Scores["severity"]
	if !ok {
		return errors.New("ask: no answer for severity")
	}

	out := struct {
		Model  string              `json:"model"`
		Noul   float64             `json:"noul_hardware_fault"`
		Choice decide.ChoiceAnswer `json:"choice_next_action"`
		Score  decide.ScoreAnswer  `json:"score_severity"`
		Usage  llmkit.Usage        `json:"usage"`
	}{
		Model:  resp.Model,
		Noul:   noul,
		Choice: choice,
		Score:  score,
		Usage:  resp.Usage,
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal result: %w", err)
	}
	fmt.Println(string(b))
	return nil
}
