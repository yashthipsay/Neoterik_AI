// popup.js - Cleaned and Enhanced

document.addEventListener("DOMContentLoaded", () => {
	console.log("✅ Popup loaded");
	initializeUI();
	setupEventListeners();
	setUIFromStorage();
});

function initializeUI() {
	document.querySelectorAll(".tab-btn").forEach((btn) => {
		btn.addEventListener("click", () => switchTab(btn.dataset.tab));
	});
}

// Add global flags for agent and cover letter generation
let isAgentRunning = false;
let isGeneratingCoverLetter = false;

function setAgentProgressState(running, message = "") {
	const agentProgress = document.getElementById("agent-progress-container");
	if (agentProgress) {
		agentProgress.classList.toggle("hidden", !running);
		const agentStatus = document.getElementById("agent-progress-status");
		if (agentStatus) agentStatus.textContent = running ? message : "";
	}
	isAgentRunning = running;
}

function setLoadingState(loading, message = "") {
	const progressContainer = document.getElementById("progress-container");
	if (progressContainer)
		progressContainer.classList.toggle("hidden", !loading);
	document.getElementById("generate-btn-text").textContent = loading
		? "Generating..."
		: "Generate Cover Letter";
	document
		.getElementById("generate-btn-spinner")
		?.classList.toggle("hidden", !loading);
	const status = document.getElementById("status-message");
	if (status) status.textContent = loading ? message : "";
	isGeneratingCoverLetter = loading;
	const generateBtn = document.getElementById("generate-btn");
	if (generateBtn) generateBtn.disabled = loading;
	// Hide status/progress if agent is running
	if (isAgentRunning && !loading) {
		if (progressContainer) progressContainer.classList.add("hidden");
		if (status) status.textContent = "";
	}
}

function setupEventListeners() {
	document
		.getElementById("signout-btn")
		?.addEventListener("click", handleSignOut);
	document
		.getElementById("cover-letter-form")
		?.addEventListener("submit", handleGenerateCoverLetter);
	document
		.getElementById("signin-btn-header")
		?.addEventListener("click", handleSignIn);
	document
		.getElementById("signin-btn-welcome")
		?.addEventListener("click", handleSignIn);
	document
		.getElementById("do-something-link")
		?.addEventListener("click", doSomething);

	chrome.storage.onChanged.addListener((changes, area) => {
		if (
			area === "local" &&
			(changes.jobSession ||
				changes.currentJobPage ||
				changes.isLoggedIn ||
				changes.user)
		) {
			setUIFromStorage();
			const jobSession = changes.jobSession?.newValue;
			if (jobSession?.agentError) {
				showError(jobSession.agentError);
				setLoadingState(false, "");
			}
			if (jobSession?.isLocked && !jobSession?.isCoverLetterGenerated) {
				window.sessionStorage.setItem("neoterik-locked-alert", "1");
			} else {
				window.sessionStorage.removeItem("neoterik-locked-alert");
			}
		}
		if (area === "local" && (changes.isLoggedIn || changes.user)) {
			const isLoggedIn = changes.isLoggedIn?.newValue;
			const user = changes.user?.newValue;
			updateUI(!!isLoggedIn, user);
		}
	});
	if (window.sessionStorage.getItem("neoterik-locked-alert") === "1") {
		alert(
			"You have already started generating a cover letter for this job. Please complete or skip before starting a new one."
		);
		window.sessionStorage.removeItem("neoterik-locked-alert");
	}
	chrome.runtime.onMessage.addListener((message) => {
		if (message.action === "loginStatusChanged") {
			chrome.storage.local.get(["isLoggedIn", "user"], (data) => {
				updateUI(data.isLoggedIn, data.user);
			});
		}
		if (message.action === "coverLetterGenerated") {
			showCoverLetterPreview(message.coverLetter);
			switchTab("preview");
			setLoadingState(false, "");
		}
		if (message.action === "coverLetterError") {
			showError(message.error || "Failed to generate cover letter");
			setLoadingState(false, "");
		}
	});
}

function switchTab(tabId) {
	document
		.querySelectorAll(".tab-btn")
		.forEach((btn) => btn.classList.remove("active"));
	document.querySelector(`[data-tab="${tabId}"]`)?.classList.add("active");
	document
		.querySelectorAll('[id$="-tab"]')
		.forEach((el) => el.classList.add("hidden"));
	document.getElementById(`${tabId}-tab`)?.classList.remove("hidden");
	if (tabId !== "generate") setLoadingState(false);
}

function checkAuthState() {
	chrome.storage.local.get(["isLoggedIn", "user"], ({ isLoggedIn, user }) => {
		console.log("🔍 Auth check:", isLoggedIn, user);
		updateUI(isLoggedIn, user);
	});
}

function updateUI(isLoggedIn, user) {
	document
		.getElementById("signin-btn-header")
		?.classList.toggle("hidden", isLoggedIn);
	document
		.getElementById("signin-btn-welcome")
		?.classList.toggle("hidden", isLoggedIn);
	document
		.getElementById("signout-btn")
		?.classList.toggle("hidden", !isLoggedIn);
	document
		.getElementById("user-info")
		?.classList.toggle("hidden", !isLoggedIn);
	document
		.getElementById("welcome-state")
		?.classList.toggle("hidden", isLoggedIn);
	document
		.getElementById("generate-tab")
		?.classList.toggle("hidden", !isLoggedIn);
	document.getElementById("loading-indicator")?.classList.add("hidden");

	if (isLoggedIn && user) {
		document.getElementById("user-name").textContent =
			user.name?.split(" ")[0] || "User";
		const avatar = document.getElementById("user-avatar");
		if (avatar && user.image) {
			avatar.src = user.image;
			avatar.style.display = "block";
		}
	}
}
 
function handleSignIn() {
	console.log("🔐 Sign in initiated");
	document.getElementById("loading-indicator")?.classList.remove("hidden");

	const width = 600,
		height = 700;
	chrome.windows.create({
		url: "http://localhost:3000/auth/signin",
		type: "popup",
		width,
		height,
		left: Math.round((screen.width - width) / 2),
		top: Math.round((screen.height - height) / 2),
	});
}

function handleSignOut() {
	chrome.storage.local.remove(
		["isLoggedIn", "user", "jobSession", "currentJobPage"],
		() => {
			updateUI(false, null);
			document
				.getElementById("welcome-state")
				?.classList.remove("hidden");
			document
				.getElementById("signin-btn-header")
				?.classList.remove("hidden");
			document
				.getElementById("signin-btn-welcome")
				?.classList.remove("hidden");
			document.getElementById("signout-btn")?.classList.add("hidden");
			document.getElementById("user-info")?.classList.add("hidden");
			document.getElementById("generate-tab")?.classList.add("hidden");
			[
				"#company_name",
				"#job_title",
				"#job_description",
				"#company_summary",
				"#company_vision",
				"#additional_notes",
				"#preferred_skills",
			].forEach((id) => {
				const field = document.querySelector(id);
				if (field) field.value = "";
			});
			document.getElementById("cover-letter-preview").innerHTML = "";
			chrome.tabs.create({
				url: "http://localhost:3000/api/auth/signout",
				active: false,
			});
		}
	);
}

function populateFieldsFromGraph() {
	chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
		const job = currentJobPage?.jobData;
		if (!job) return;

		const fill = (id, val) => {
			const field = document.querySelector(id);
			if (field && val) field.value = val;
		};

		fill("#company_name", job.company_name);
		fill("#job_title", job.job_title);
		fill("#job_description", job.job_description);
		fill("#company_summary", job.company_summary);
		fill("#company_vision", job.company_vision);
		fill("#additional_notes", job.additional_notes);

		const skills = [
			...(job.preferred_qualifications || []),
			...(job.skillset || []),
		];
		fill("#preferred_skills", skills.join(", "));
	});
}

async function handleGenerateCoverLetter(e) {
	e.preventDefault();
	if (isGeneratingCoverLetter) return; // Prevent double submit
	const { jobSession } = await chrome.storage.local.get("jobSession");
	await chrome.storage.local.set({
		jobSession: {
			...jobSession,
			isUserConfirmed: true,
			isCoverLetterGenerating: true,
			coverLetterError: null, // <-- Set flag here!
		},
	});
	const data = await getJobDataForCoverLetter();
	chrome.runtime.sendMessage({ action: "generateCoverLetter", data });
	setLoadingState(
		true,
		"⏳ Our AI assistant is preparing your personalized cover letter..."
	);
}
// try {
//     const data = await getJobDataForCoverLetter();
//     console.log("📨 Submitting data:", data);
//     const res = await fetch("http://localhost:8000/generate-cover-letter", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(data),
//     });
//     const result = await res.json();
//     setLoadingState(false);
//     if (res.ok && result.cover_letter) {
//         await chrome.storage.local.get("jobSession", ({ jobSession }) => {
//             chrome.storage.local.set({
//                 jobSession: {
//                     ...jobSession,
//                     isCoverLetterGenerated: true,
//                     isCoverLetterGenerating: false // <-- Clear flag on success
//                 }
//             });
//         });
//         showCoverLetterPreview(result.cover_letter);
//         [
//             "#company_name",
//             "#job_title",
//             "#job_description",
//             "#company_summary",
//             "#company_vision",
//             "#additional_notes",
//             "#preferred_skills",
//         ].forEach((id) => {
//             const field = document.querySelector(id);
//             if (field) field.value = "";
//         });
//         switchTab("preview");
//         chrome.runtime.sendMessage({ action: "coverLetterGeneratedCleanup" });
//     } else {
//         // Clear flag on error
//         await chrome.storage.local.get("jobSession", ({ jobSession }) => {
//             chrome.storage.local.set({
//                 jobSession: {
//                     ...jobSession,
//                     isCoverLetterGenerating: false
//                 }
//             });
//         });
//         showError(result.error || "Failed to generate cover letter");
//     }
// } catch (err) {
//     console.error("❌ Generation error:", err);
//     await chrome.storage.local.get("jobSession", ({ jobSession }) => {
//         chrome.storage.local.set({
//             jobSession: {
//                 ...jobSession,
//                 isCoverLetterGenerating: false
//             }
//         });
//     });
//     showError("Network error. Please try again.");
//     setLoadingState(false);
// }
//}

// async function handleGenerateCoverLetter(e) {
// 	e.preventDefault();
// 	if (isGeneratingCoverLetter) return; // Prevent double submit
// 	const { jobSession } = await chrome.storage.local.get("jobSession");
// 	await chrome.storage.local.set({ jobSession: { ...jobSession, isUserConfirmed: true } });
// 	setLoadingState(true, "⏳ Our AI assistant is preparing your personalized cover letter...");
// 	try {
// 		const data = await getJobDataForCoverLetter();
// 		console.log("📨 Submitting data:", data);
// 		const res = await fetch("http://localhost:8000/generate-cover-letter", {
// 			method: "POST",
// 			headers: { "Content-Type": "application/json" },
// 			body: JSON.stringify(data),
// 		});
// 		const result = await res.json();
// 		setLoadingState(false);
// 		if (res.ok && result.cover_letter) {
// 			await chrome.storage.local.get("jobSession", ({ jobSession }) => {
// 				chrome.storage.local.set({ jobSession: { ...jobSession, isCoverLetterGenerated: true } });
// 			});
// 			showCoverLetterPreview(result.cover_letter);
// 			// Clear all fields on generate tab
// 			[
// 				"#company_name",
// 				"#job_title",
// 				"#job_description",
// 				"#company_summary",
// 				"#company_vision",
// 				"#additional_notes",
// 				"#preferred_skills",
// 			].forEach((id) => {
// 				const field = document.querySelector(id);
// 				if (field) field.value = "";
// 			});
// 			// Switch to preview tab
// 			switchTab("preview");
// 			// Clear jobSession and currentJobPage after short delay
// 			chrome.runtime.sendMessage({ action: "coverLetterGeneratedCleanup" });
// 		} else {
// 			showError(result.error || "Failed to generate cover letter");
// 		}
// 	} catch (err) {
// 		console.error("❌ Generation error:", err);
// 		showError("Network error. Please try again.");
// 		setLoadingState(false);
// 	}
// }

function showCoverLetterPreview(text) {
	document.getElementById(
		"cover-letter-preview"
	).innerHTML = `<div style="white-space: pre-wrap; line-height: 1.6;">${text}</div>`;
}

function showError(msg) {
	alert(msg);
}

function copyToClipboard() {
	const text = document.getElementById("cover-letter-preview")?.textContent;
	if (text) navigator.clipboard.writeText(text).then(() => alert("Copied!"));
}

function editCoverLetter() {
	switchTab("generate");
}

function saveCoverLetter() {
	alert("Cover letter saved to your library!");
}

async function getJobDataForCoverLetter() {
	return new Promise((resolve) => {
		chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
			const job = currentJobPage?.jobData || {};
			resolve({
				job_title: job.job_title || "",
				hiring_company: job.company_name || "",
				applicant_name: "Yash Thipsay",
				job_description: job.job_description || "",
				preferred_qualifications: [
					...(job.preferred_qualifications || []),
					...(job.skillset || []),
				].join("; "),
				company_culture_notes: `${job.company_vision || ""}\n${
					job.additional_notes || ""
				}`,
				github_username: "yashthipsay",
				desired_tone:
					document.getElementById("desired_tone")?.value ||
					"professional",
				company_url: "",
			});
		});
	});
}

function showGenerateTabAndPopulate() {
	switchTab("generate");
	populateFieldsFromGraph();
	setLoadingState(false, ""); // Ensure button is not loading
}

// Restore original authentication UI/logic
function setUIFromStorage() {
	chrome.storage.local.get(
		["isLoggedIn", "user", "jobSession", "currentJobPage"],
		({ isLoggedIn, user, jobSession, currentJobPage }) => {
			// Restore authentication UI
			updateUI(!!isLoggedIn, user);

			// Job/agent/cover letter state
			const isAgentInProgress = jobSession?.isAgentInProgress;
			const isAgentFinished = jobSession?.isAgentFinished;
			const isCoverLetterGenerating = jobSession?.isCoverLetterGenerating;
			const isCoverLetterGenerated = jobSession?.isCoverLetterGenerated;
			const jobData = currentJobPage?.jobData;

			// 1. Show preview if cover letter is generated
			if (isCoverLetterGenerated && jobSession?.coverLetter) {
				showCoverLetterPreview(jobSession.coverLetter);
				switchTab("preview");
				setLoadingState(false, "");
				return;
			}

			// 2. Show loading if generating
			if (isCoverLetterGenerating && jobData) {
				populateFieldsFromGraph();
				setLoadingState(
					true,
					"⏳ Our AI assistant is preparing your personalized cover letter..."
				);
				return;
			}

			if (isAgentInProgress) {
				setAgentProgressState(
					true,
					"NeoterikAi's Agent Fetching Job Details.."
				);
				setLoadingState(false, "");
				switchTab("generate");
				return;
			} else {
				setAgentProgressState(false);
			}

			if (isAgentFinished && jobData) {
				showGenerateTabAndPopulate();
				return;
			}

			if (jobData) {
				populateFieldsFromGraph();
				switchTab("generate");
				setLoadingState(false, "");
				return;
			}
			// 6. Default: clear fields in generate tab if no jobData or after cover letter generated
			clearGenerateTabFields();
			setLoadingState(false, "");
		})
}

// Helper to clear all fields in generate tab
function clearGenerateTabFields() {
    [
        "#company_name",
        "#job_title",
        "#job_description",
        "#company_summary",
        "#company_vision",
        "#additional_notes",
        "#preferred_skills",
    ].forEach((id) => {
        const field = document.querySelector(id);
        if (field) field.value = "";
    });
}


