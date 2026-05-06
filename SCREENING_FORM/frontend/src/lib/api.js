import axios from "axios";
import { API_URL } from "../constants";

export async function registerUser(prolificId) {
  return axios.post(`${API_URL}/users`, {
    prolific_id: prolificId,
  });
}

export async function saveDraft(prolificId, draft) {
  return axios.put(`${API_URL}/draft/${encodeURIComponent(prolificId)}`, draft);
}

export async function loadDraft(prolificId) {
  try {
    const { data } = await axios.get(
      `${API_URL}/draft/${encodeURIComponent(prolificId)}`,
    );
    return data;
  } catch (err) {
    if (err.response?.status === 404) return null;
    throw err;
  }
}

export async function submitFinal(payload) {
  return axios.post(`${API_URL}/submit`, payload);
}
