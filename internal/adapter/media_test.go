package adapter

import "testing"

func TestDataURLMediaType(t *testing.T) {
	cases := []struct {
		url    string
		want   string
		wantOK bool
	}{
		{"data:image/png;base64,AAAA", "image/png", true},
		{"data:image/bmp,Qk0=", "image/bmp", true},
		{"data:image/png; x=y;base64,AAAA", "image/png", true},
		{"DATA:image/bmp;base64,Qk0=", "image/bmp", true},
		{"Data:IMAGE/PNG;base64,AAAA", "IMAGE/PNG", true},
		{"data:;base64,AAAA", "", true},
		{"data:,hello", "", true},
		{"data:image/png", "", true},
		{"data:", "", true},
		{"https://example.test/a.png", "", false},
		{"gs://bucket/clip.wav", "", false},
		{" data:image/png;base64,AAAA", "", false},
		{"dat", "", false},
		{"", "", false},
	}
	for _, c := range cases {
		got, ok := DataURLMediaType(c.url)
		if got != c.want || ok != c.wantOK {
			t.Errorf("DataURLMediaType(%q) = %q, %v; want %q, %v", c.url, got, ok, c.want, c.wantOK)
		}
	}
}
