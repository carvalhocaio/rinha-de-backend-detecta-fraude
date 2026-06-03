import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    rampa: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "10s", target: 20 },
        { duration: "20s", target: 50 },
        { duration: "10s", target: 0 },
      ],
    },
  },
  thresholds: { http_req_duration: ["p(99)<2000"] },
};

const payload = JSON.stringify({
  id: "tx-1329056812",
  transaction: {
    amount: 41.12,
    installments: 2,
    requested_at: "2026-03-11T18:45:53Z",
  },
  customer: {
    avg_amount: 82.24,
    tx_count_24h: 3,
    known_merchants: ["MERC-003", "MERC-016"],
  },
  merchant: { id: "MERC-016", mcc: "5411", avg_amount: 60.25 },
  terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
  last_transaction: null,
});
const params = { headers: { "Content-Type": "application/json" } };

export default function () {
  const res = http.post("http://localhost:9999/fraud-score", payload, params);
  check(res, { "status 200": (r) => r.status === 200 });
}
