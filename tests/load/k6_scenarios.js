/**
 * k6 load test — RAG Platform SLA verification
 *
 * Profiles:
 *   profile50:  50 concurrent, 5 min  → 95% ≤ 10s
 *   profile99:  99 concurrent, 5 min  → queue tier2
 *   profile499: 499 concurrent, 3 min → queue tier3
 *
 * Run: k6 run --env TARGET=http://gateway:8081 k6_scenarios.js
 */
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const errorRate = new Rate('errors');
const queryDuration = new Trend('query_duration_ms', true);

const TARGET = __ENV.TARGET || 'http://localhost:8081';

export const options = {
  scenarios: {
    profile50: {
      executor: 'constant-vus',
      vus: 50,
      duration: '5m',
      tags: { profile: 'tier1' },
    },
    profile99: {
      executor: 'constant-vus',
      vus: 99,
      duration: '5m',
      startTime: '6m',
      tags: { profile: 'tier2' },
    },
    profile499: {
      executor: 'constant-vus',
      vus: 499,
      duration: '3m',
      startTime: '12m',
      tags: { profile: 'tier3' },
    },
  },
  thresholds: {
    // SLA tier1: ≤50 users, p(95) ≤ 10s
    'query_duration_ms{profile:tier1}': ['p(95)<10000'],
    // SLA tier2: 51-99 users, p(95) ≤ 30s
    'query_duration_ms{profile:tier2}': ['p(95)<30000'],
    // SLA tier3: 100-499 users, p(95) ≤ 90s
    'query_duration_ms{profile:tier3}': ['p(95)<90000'],
    errors: ['rate<0.01'],
  },
};

// The demo handbook's subject area, matching k6_smoke.js and the corpus this
// repository ships.
const QUERIES = [
  'how many visitors may one employee escort at a time?',
  'what is the daily allowance for a trip outside Europe?',
  'when does a change to an approved procedure take effect?',
  'how are near misses reported?',
  'how long are backups kept before recovery is no longer possible?',
];

export default function () {
  const query = QUERIES[Math.floor(Math.random() * QUERIES.length)];
  const payload = JSON.stringify({ text: query });
  const params = {
    headers: { 'Content-Type': 'application/json' },
    timeout: '35s',
  };

  const start = Date.now();
  const res = http.post(`${TARGET}/query`, payload, params);
  const duration = Date.now() - start;

  queryDuration.add(duration);

  const ok = check(res, {
    'status 200': (r) => r.status === 200,
    'has answer field': (r) => {
      try {
        const body = JSON.parse(r.body);
        return 'text' in body || 'refused' in body;
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!ok);
  sleep(0.5);
}
