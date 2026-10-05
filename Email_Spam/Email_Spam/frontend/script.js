const API_URL = "http://127.0.0.1:8000/predict";
const emailInput = document.querySelector("#email-text");
const checkButton = document.querySelector("#check-button");
const characterCount = document.querySelector("#character-count");
const errorMessage = document.querySelector("#error-message");
const resultPanel = document.querySelector("#result-panel");
const resultIcon = document.querySelector(".result-icon");
const resultTitle = document.querySelector("#result-title");
const resultDescription = document.querySelector("#result-description");
const confidenceWrap = document.querySelector(".confidence-wrap");
const confidenceValue = document.querySelector("#confidence-value");
const confidenceBar = document.querySelector("#confidence-bar");

const examples = {
	spam: "Congratulations! You have won a free iPhone. Click here to claim your prize now!",
	safe: "Hi, please send me the project report before tomorrow's meeting. Thanks."
};

function updateCharacterCount() {
	const length = emailInput.value.length;
	characterCount.textContent = `${length} character${length === 1 ? "" : "s"}`;
}

function showError(message) {
	errorMessage.textContent = message;
	errorMessage.classList.add("visible");
}

function clearError() {
	errorMessage.textContent = "";
	errorMessage.classList.remove("visible");
}

function showResult(data) {
	const isSpam = data.result === "Spam" || Number(data.prediction) === 1;
	const confidence = data.confidence === null ? null : Number(data.confidence);

	resultPanel.classList.remove("empty-state", "spam-result", "safe-result");
	resultPanel.classList.add(isSpam ? "spam-result" : "safe-result");
	resultIcon.innerHTML = isSpam ? "!" : "&#10003;";
	resultTitle.innerHTML = isSpam ? "This looks like<br>spam." : "This looks<br>legitimate.";
	resultDescription.textContent = isSpam
		? "The model found patterns commonly associated with unwanted or deceptive messages."
		: "The model did not find strong spam signals in this message.";

	if (confidence !== null && Number.isFinite(confidence)) {
		confidenceWrap.hidden = false;
		confidenceValue.textContent = `${confidence.toFixed(1)}%`;
		confidenceBar.style.width = `${Math.min(Math.max(confidence, 0), 100)}%`;
	} else {
		confidenceWrap.hidden = true;
	}
}

async function checkEmail() {
	const emailText = emailInput.value.trim();
	clearError();

	if (!emailText) {
		showError("Add an email message before checking.");
		emailInput.focus();
		return;
	}

	checkButton.disabled = true;
	checkButton.classList.add("loading");
	checkButton.querySelector("span:first-child").textContent = "Checking...";

	try {
		const response = await fetch(API_URL, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email_text: emailText })
		});
		const data = await response.json();

		if (!response.ok) {
			throw new Error(data.detail || "The API could not check this message.");
		}

		showResult(data);
	} catch (error) {
		showError(error.message.includes("fetch")
			? "Could not reach the API. Start the FastAPI server on port 8000 and try again."
			: error.message);
	} finally {
		checkButton.disabled = false;
		checkButton.classList.remove("loading");
		checkButton.querySelector("span:first-child").textContent = "Check email";
	}
}

emailInput.addEventListener("input", updateCharacterCount);
checkButton.addEventListener("click", checkEmail);
emailInput.addEventListener("keydown", (event) => {
	if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
		checkEmail();
	}
});

document.querySelectorAll(".example-button").forEach((button) => {
	button.addEventListener("click", () => {
		emailInput.value = examples[button.dataset.example];
		updateCharacterCount();
		clearError();
		emailInput.focus();
	});
});

updateCharacterCount();
