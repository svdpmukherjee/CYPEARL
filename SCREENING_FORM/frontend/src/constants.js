export const API_URL = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? "https://prolific-screening-form.onrender.com"
    : "/api")
).replace(/\/$/, "");

export const FREQUENCY_OPTIONS = ["Daily", "Weekly", "Monthly", "Rarely"];

export const JOB_CLUSTERS = [
  "Finance / Accounts Payable",
  "IT Support / Helpdesk",
  "HR / People Operations",
  "Sales / Business Development",
  "Operations / Logistics",
  "Customer Service / Client Support",
  "Marketing / Communications",
  "Procurement / Purchasing",
  "Administrative / Executive Support",
  "Compliance / Risk / Audit",
];

export const MANDATORY_EMAILS = 5;
export const REQUIRED_GENERIC = 3;
export const REQUIRED_SUSPICIOUS = 5;
export const BONUS_PER_EMAIL_PENCE = 5;
export const MAX_BONUS_EMAILS = 0.5;

export const ADDITIONAL_SENDER_OPTIONS = [
  "HR / People Operations",
  "IT / Helpdesk",
  "Finance / Accounts",
  "External vendors / Suppliers",
  "Clients / Customers",
  "Other internal team",
  "None of the above",
];
