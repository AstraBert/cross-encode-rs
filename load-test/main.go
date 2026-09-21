package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const QUERY string = "How many people live in Berlin?"

var DOCUMENTS [10]string = [10]string{
	"Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
	"Berlin is well known for its museums.",
	"In 2014, the city state Berlin had 37,368 live births (+6.6%), a record number since 1991.",
	"The urban area of Berlin comprised about 4.1 million people in 2014, making it the seventh most populous urban area in the European Union.",
	"The city of Paris had a population of 2,165,423 people within its administrative city limits as of January 1, 2019",
	"An estimated 300,000-420,000 Muslims reside in Berlin, making up about 8-11 percent of the population.",
	"Berlin is subdivided into 12 boroughs or districts (Bezirke).",
	"In 2015, the total labour force in Berlin was 1.85 million.",
	"In 2013 around 600,000 Berliners were registered in one of the more than 2,300 sport and fitness clubs.",
	"Berlin has a yearly total of about 135 million day visitors, which puts it in third place among the most-visited city destinations in the European Union.",
}

type RerankRequest struct {
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	ReturnDocuments bool     `json:"return_documents"`
}

// FailureCounts aggregates why requests failed, since printing one line per
// failure would flood stdout at high concurrency. Reasons are bucketed by a
// short label (e.g. "http 503" or the connection error string) so patterns
// are still visible in the final summary.
type FailureCounts struct {
	mu     sync.Mutex
	counts map[string]int
}

func newFailureCounts() *FailureCounts {
	return &FailureCounts{counts: make(map[string]int)}
}

func (f *FailureCounts) record(reason string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.counts[reason]++
}

func (f *FailureCounts) print() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.counts) == 0 {
		return
	}

	type kv struct {
		reason string
		count  int
	}
	sorted := make([]kv, 0, len(f.counts))
	for reason, count := range f.counts {
		sorted = append(sorted, kv{reason, count})
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].count > sorted[j].count })

	fmt.Println("Failure reasons:")
	for _, entry := range sorted {
		fmt.Printf("  %5d  %s\n", entry.count, entry.reason)
	}
}

func sendPostRequest(endpoint string, client *http.Client, successTime *atomic.Int64, failed *atomic.Int32, success *atomic.Int32, failures *FailureCounts) {

	requestJSON := RerankRequest{Query: QUERY, Documents: DOCUMENTS[:], ReturnDocuments: true}
	requestBody, err := json.Marshal(requestJSON)
	if err != nil {
		failures.record(fmt.Sprintf("request marshal error: %s", err.Error()))
		failed.Add(1)
		return
	}

	request, err := http.NewRequest("POST", endpoint, bytes.NewReader(requestBody))
	if err != nil {
		failures.record(fmt.Sprintf("request build error: %s", err.Error()))
		failed.Add(1)
		return
	}
	request.Header.Set("Content-Type", "application/json")

	start := time.Now()
	response, err := client.Do(request)
	if err != nil {
		// Connection-level failure (refused, reset, timeout, too many open
		// files, ...) — this never reached the server, so it won't show up
		// in the server's logs either.
		failures.record(fmt.Sprintf("connection error: %s", err.Error()))
		failed.Add(1)
		return
	}
	defer response.Body.Close()

	elapsed := time.Since(start).Milliseconds()

	if response.StatusCode == 200 {
		successTime.Add(elapsed)
		success.Add(1)
		return
	}

	body, readErr := io.ReadAll(response.Body)
	if readErr != nil {
		failures.record(fmt.Sprintf("http %d (body unreadable: %s)", response.StatusCode, readErr.Error()))
	} else {
		failures.record(fmt.Sprintf("http %d: %s", response.StatusCode, string(body)))
	}
	failed.Add(1)
}

func main() {
	args := os.Args
	if len(args) != 3 {
		log.Fatalf("Expecting exactly two arguments from command line")
	}

	howMany, err := strconv.Atoi(args[1])
	if err != nil {
		log.Fatalf("Expecting the second argument to represent how many requests have to be sent")
	}

	maxConcurrent := 1000
	semaphore := make(chan struct{}, maxConcurrent)

	transport := &http.Transport{
		MaxIdleConns:        maxConcurrent,
		MaxIdleConnsPerHost: maxConcurrent,
		MaxConnsPerHost:     maxConcurrent,
		IdleConnTimeout:     90 * time.Second,
	}

	client := &http.Client{
		Timeout:   10 * time.Second,
		Transport: transport,
	}

	var wg sync.WaitGroup
	var successTime atomic.Int64
	var failed atomic.Int32
	var success atomic.Int32
	failures := newFailureCounts()

	startTime := time.Now()

	for range howMany {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore

		go func() {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore
			sendPostRequest(args[2], client, &successTime, &failed, &success, failures)
		}()
	}

	wg.Wait()
	totalDuration := time.Since(startTime)

	successCount := int(success.Load())
	failedCount := int(failed.Load())
	totalTime := successTime.Load()

	var avgTime float64
	if successCount > 0 {
		avgTime = float64(totalTime) / float64(successCount)
	}

	fmt.Printf("Total requests: %d\n", howMany)
	fmt.Printf("Successful requests: %d\n", successCount)
	fmt.Printf("Failed requests: %d\n", failedCount)
	fmt.Printf("Success rate: %.2f%%\n", float64(successCount)/float64(howMany)*100)
	fmt.Printf("Average response time: %.2f ms\n", avgTime)
	fmt.Printf("Total test duration: %v\n", totalDuration)
	fmt.Printf("Requests per second: %.2f\n", float64(howMany)/totalDuration.Seconds())
	failures.print()
}
