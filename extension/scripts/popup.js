// Enhanced popup script with modern UI interactions
document.addEventListener("DOMContentLoaded", function () {
	console.log("Enhanced popup DOM loaded.");

	// Initialize UI state
	initializeUI();
	setupEventListeners();
	checkAuthState();
	populateFieldsFromGraph();
});

function initializeUI() {
	// Set up tab navigation
	const tabButtons = document.querySelectorAll(".tab-btn");
	const tabContents = document.querySelectorAll('[id$="-tab"]');

	tabButtons.forEach((button) => {
		button.addEventListener("click", () => {
			const tabId = button.dataset.tab;
			switchTab(tabId);
		});
	});
}

function setupEventListeners() {
	// Sign out button
	const signoutButton = document.getElementById("signout-btn");
	if (signoutButton) {
		signoutButton.addEventListener("click", handleSignOut);
	}

	// Cover letter form
const coverLetterForm = document.getElementById("cover-letter-form");
if (coverLetterForm) {
    coverLetterForm.addEventListener("submit", handleGenerateCoverLetter);
}

	const signinBtnHeader = document.getElementById("signin-btn-header");
	if (signinBtnHeader) {
		signinBtnHeader.addEventListener("click", handleSignIn);
	}
	const signinBtnWelcome = document.getElementById("signin-btn-welcome");
	if (signinBtnWelcome) {
		signinBtnWelcome.addEventListener("click", handleSignIn);
	}

	// filepath: extension/scripts/popup.js
	const doSomethingLink = document.getElementById("do-something-link");
	if (doSomethingLink) {
		doSomethingLink.addEventListener("click", doSomething);
	}

	// Listen for storage changes
	chrome.storage.onChanged.addListener((changes, area) => {
		if (area === "local" && (changes.isLoggedIn || changes.user)) {
			checkAuthState();
		}
	});

	// Listen for messages from background script
	chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
		if (message.action === "loginStatusChanged") {
			checkAuthState();
		}
	});
}

function switchTab(tabId) {
	// Update tab buttons
	document.querySelectorAll(".tab-btn").forEach((btn) => {
		btn.classList.remove("active");
	});
	document.querySelector(`[data-tab="${tabId}"]`).classList.add("active");

	// Update tab content
	document.querySelectorAll('[id$="-tab"]').forEach((content) => {
		content.classList.add("hidden");
	});

	const targetTab = document.getElementById(`${tabId}-tab`);
	if (targetTab) {
		targetTab.classList.remove("hidden");
		targetTab.classList.add("animate-fadeIn");
	}

	if (tabId !== "generate") setLoadingState(false);
}

function checkAuthState() {
	chrome.storage.local.get(["isLoggedIn", "user"], (result) => {
		console.log("Auth state:", result);
		updateUI(result.isLoggedIn, result.user);
	});
}

function updateUI(isLoggedIn, user) {
	const signinButtonHeader = document.getElementById("signin-btn-header");
	const signinButtonWelcome = document.getElementById("signin-btn-welcome");
	const signoutButton = document.getElementById("signout-btn");
	const userInfo = document.getElementById("user-info");
	const welcomeState = document.getElementById("welcome-state");
	const generateTab = document.getElementById("generate-tab");
	const loadingIndicator = document.getElementById("loading-indicator");

	// Hide loading indicator
	if (loadingIndicator) {
		loadingIndicator.classList.add("hidden");
	}

	if (isLoggedIn && user) {
		// Authenticated state
		if (signinButtonHeader) signinButtonHeader.classList.add("hidden");
		if (signinButtonWelcome) signinButtonWelcome.classList.add("hidden");
		if (signoutButton) signoutButton.classList.remove("hidden");
		if (welcomeState) welcomeState.classList.add("hidden");
		if (generateTab) generateTab.classList.remove("hidden");

		if (userInfo) {
			userInfo.classList.remove("hidden");
			const userName = document.getElementById("user-name");
			const userAvatar = document.getElementById("user-avatar");

			if (userName) {
				userName.textContent = user.name?.split(" ")[0] || "User";
			}

			if (userAvatar && user.image) {
				userAvatar.src = user.image;
				userAvatar.style.display = "block";
			}
		}
	} else {
		// Unauthenticated state
		if (signinButtonHeader) signinButtonHeader.classList.remove("hidden");
		if (signinButtonWelcome) signinButtonWelcome.classList.remove("hidden");
		if (signoutButton) signoutButton.classList.add("hidden");
		if (userInfo) userInfo.classList.add("hidden");
		if (welcomeState) welcomeState.classList.remove("hidden");
		if (generateTab) generateTab.classList.add("hidden");
	}
}

function handleSignIn() {
	console.log("Sign in button clicked");

	// Show loading state
	const loadingIndicator = document.getElementById("loading-indicator");
	if (loadingIndicator) {
		loadingIndicator.classList.remove("hidden");
	}

	// Check if running in extension context
	if (typeof chrome !== "undefined" && chrome.windows) {
		// Extension context - open popup window
		const width = 600;
		const height = 700;
		const left = Math.round((screen.width - width) / 2);
		const top = Math.round((screen.height - height) / 2);

		chrome.windows.create({
			url: "http://localhost:3000/auth/signin",
			type: "popup",
			width: width,
			height: height,
			left: left,
			top: top,
		});
	} else {
		// Web context - redirect
		window.location.href = "http://localhost:3000/auth/signin";
	}
}

function handleSignOut() {
	console.log("Sign out button clicked");

	chrome.storage.local.remove(["isLoggedIn", "user"], () => {
		console.log("Cleared local auth state");
		updateUI(false, null);

		// Clear server session
		if (typeof chrome !== "undefined" && chrome.tabs) {
			chrome.tabs.create({
				url: "http://localhost:3000/api/auth/signout",
				active: false,
			});
		}
	});
}

function populateFieldsFromGraph() {
	chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
		if (currentJobPage && currentJobPage.jobData) {
			const job = currentJobPage.jobData;
			console.log("Filling fields from parsed job data:", job);

			const fillField = (selector, value) => {
				const field = document.querySelector(selector);
				if (field && value) field.value = value;
			};

			fillField("#company_name", job.company_name);
			fillField("#job_title", job.job_title);
			fillField("#job_description", job.job_description);
			fillField("#company_summary", job.company_summary);
			fillField("#company_vision", job.company_vision);
			fillField("#additional_notes", job.additional_notes);

			// ✅ Combine preferred qualifications + skillset into one field
			const mergedSkills = [
				...(job.preferred_qualifications || []),
				...(job.skillset || []),
			];
			fillField("#preferred_skills", mergedSkills.join(", "));
		}
	});
}

// ---------------- Unified Data Extractor ---------------- //
async function getFinalJobData() {
	return new Promise((resolve) => {
		chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
			const job = currentJobPage?.jobData || {};
			const get = (id, fallback = "") => {
				const el = document.getElementById(id);
				return el?.value.trim() || fallback;
			};

			const data = {
				companyName: get("company_name", job.company_name),
				jobTitle: get("job_title", job.job_title),
				jobDescription: get("job_description", job.job_description),
				companySummary: get("company_summary", job.company_summary),
				companyVision: get("company_vision", job.company_vision),
				additionalNotes: get("additional_notes", job.additional_notes),
				resumeHighlights: get("resume_highlights"),
				skillsAndQualifications: get(
					"preferred_skills",
					[
						...(job.preferred_qualifications || []),
						...(job.skillset || []),
					].join(", ")
				),
			};

			resolve(data);
		});
	});
}

function setLoadingState(isLoading) {
    console.log(`🔄 Setting loading state: ${isLoading}`);
    const progressContainer = document.getElementById("progress-container");
    const generateBtnText = document.getElementById("generate-btn-text");
    const generateBtnSpinner = document.getElementById("generate-btn-spinner");
    
    if (progressContainer) progressContainer.classList.toggle("hidden", !isLoading);
    if (generateBtnText) generateBtnText.textContent = isLoading ? "Generating..." : "Generate Cover Letter";
    if (generateBtnSpinner) generateBtnSpinner.classList.toggle("hidden", !isLoading);
}

// async function getJobDataForCoverLetter() {
//     console.log("🔍 Fetching job data for cover letter");
//     return new Promise((resolve) => {
//         chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
//             const job = currentJobPage?.jobData || {};
//             console.log("📋 Current job data:", job);
            
//             const get = (id, fallback = "") => {
//                 const el = document.getElementById(id);
//                 return el?.value?.trim() || fallback;
//             };

//             const data = {
//                 job_title: get("job_title", job.job_title || ""),
//                 hiring_company: get("company_name", job.company_name || ""),
//                 applicant_name: get("applicant_name", ""), // Add a field for this in your form
//                 job_description: get("job_description", job.job_description || ""),
//                 preferred_qualifications: get("preferred_skills", job.preferred_qualifications || ""),
//                 company_culture_notes: get("company_culture_notes", job.company_culture_notes || ""),
//                 github_username: get("github_username", job.github_username || ""),
//                 company_url: get("company_url", job.company_url || "")
//             };

//             console.log("✨ Prepared cover letter data:", data);
//             resolve(data);
//         });
//     });
// }



// ---------------- Submit Handler ---------------- //
async function handleGenerateCoverLetter(event) {
    event.preventDefault();
    console.log("🎯 Generate cover letter form submitted");
    
    setLoadingState(true);
    try {
        const data = await getJobDataForCoverLetter();
        console.log("📦 Sending to /generate-cover-letter:", data);

        const response = await fetch("http://localhost:8000/generate-cover-letter", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log("📩 Received response:", result);
        
        setLoadingState(false);

        if (response.ok && result.cover_letter) {
            console.log("✅ Successfully generated cover letter");
            showCoverLetterPreview(result.cover_letter);
            switchTab("preview");
        } else {
            console.error("❌ Failed to generate cover letter:", result.error);
            showError(result.error || "Failed to generate cover letter");
        }
    } catch (err) {
        console.error("❌ Network or server error:", err);
        setLoadingState(false);
        showError("Network or server error. Please try again.");
    }
}


async function getJobDataForCoverLetter() {
    console.log("🔍 Fetching job data for cover letter");
    return new Promise((resolve) => {
        chrome.storage.local.get(["currentJobPage"], ({ currentJobPage }) => {
            const job = currentJobPage?.jobData || {};
            console.log("📋 Current job data:", job);
            
            // Join skillset and preferred qualifications into a single string
            const qualifications = [
                ...(job.preferred_qualifications || []),
                ...(job.skillset || [])
            ].join("; ");

            // Format culture notes from additional_notes
            const cultureNotes = job.additional_notes || job.company_culture_notes || "";

            const data = {
                job_title: job.job_title || "",
                hiring_company: job.company_name || "",
                applicant_name: "Yash Thipsay", // Hardcoded name
                job_description: job.job_description || "",
                preferred_qualifications: qualifications,
                company_culture_notes: `${job.company_vision || ""}\n${cultureNotes}`,
                github_username: "yashthipsay", // Hardcoded GitHub username
                desired_tone: document.getElementById("desired_tone")?.value || "professional", // Get selected tone
                company_url: "" // Optional field
            };

            console.log("✨ Prepared cover letter data:", data);
            resolve(data);
        });
    });
}
// Utility to show error messages
function showError(msg) {
    alert(msg); // Or display in a UI element
}

// ---------------- Cover Letter Generation ---------------- //
function showGenerationProgress() {
	const progressContainer = document.getElementById("progress-container");
	const generateBtnText = document.getElementById("generate-btn-text");
	const generateBtnSpinner = document.getElementById("generate-btn-spinner");
	const progressFill = document.getElementById("progress-fill");
	const progressPercent = document.getElementById("progress-percent");

	progressContainer?.classList.remove("hidden");
	if (generateBtnText) generateBtnText.textContent = "Generating...";
	generateBtnSpinner?.classList.remove("hidden");

	let progress = 0;
	const interval = setInterval(() => {
		progress += Math.random() * 15;
		if (progress > 90) progress = 90;
		if (progressFill) progressFill.style.width = `${progress}%`;
		if (progressPercent)
			progressPercent.textContent = `${Math.round(progress)}%`;
		if (progress >= 90) clearInterval(interval);
	}, 200);
}

function hideGenerationProgress() {
	document.getElementById("progress-container")?.classList.add("hidden");
	const btnText = document.getElementById("generate-btn-text");
	if (btnText) btnText.textContent = "Generate Cover Letter";
	document.getElementById("generate-btn-spinner")?.classList.add("hidden");
}

function simulateGeneration(data) {
	setTimeout(() => {
		const mockCoverLetter = generateMockCoverLetter(data);
		document.getElementById("progress-fill").style.width = "100%";
		document.getElementById("progress-percent").textContent = "100%";

		setTimeout(() => {
			hideGenerationProgress();
			showCoverLetterPreview(mockCoverLetter);
			switchTab("preview");
		}, 500);
	}, 3000);
}

function generateMockCoverLetter(data) {
	return `Dear Hiring Manager,

I am writing to express my strong interest in the ${
		data.jobTitle
	} position at ${data.companyName}.

Here’s why I’m a great fit:
${
	data.skillsAndQualifications
		? "• " + data.skillsAndQualifications.split(", ").join("\n• ") + "\n"
		: ""
}
${
	data.resumeHighlights
		? "Resume Highlights:\n" + data.resumeHighlights + "\n"
		: ""
}
${data.companySummary ? "Company Summary:\n" + data.companySummary + "\n" : ""}
${data.companyVision ? "Company Vision:\n" + data.companyVision + "\n" : ""}
${
	data.additionalNotes
		? "Additional Notes:\n" + data.additionalNotes + "\n"
		: ""
}

Thank you for considering my application.

Sincerely,
[Your Name]`;
}

function showCoverLetterPreview(coverLetter) {
    console.log("📄 Showing cover letter preview");
    const preview = document.getElementById("cover-letter-preview");
    if (preview) {
        preview.innerHTML = `<div style="white-space: pre-wrap; line-height: 1.6;">${coverLetter}</div>`;
    }
}

// ---------------- Utilities ---------------- //
function copyToClipboard() {
	const preview = document.getElementById("cover-letter-preview");
	if (preview) {
		const text = preview.textContent;
		navigator.clipboard.writeText(text).then(() => {
			const button = event.target;
			const original = button.innerHTML;
			button.innerHTML = "<span>✓</span> Copied!";
			button.style.background = "var(--success-color)";
			button.style.color = "white";

			setTimeout(() => {
				button.innerHTML = original;
				button.style.background = "";
				button.style.color = "";
			}, 2000);
		});
	}
}

function editCoverLetter() {
	switchTab("generate");
}

function saveCoverLetter() {
	alert("Cover letter saved to your library!");
}
// function handleGenerateCoverLetter(event) {
// 	event.preventDefault();

// 	console.log("Generate cover letter form submitted");

// 	// Get form data
// 	const formData = new FormData(event.target);
// 	const inputs = event.target.querySelectorAll("input, textarea");
// 	const data = {};

// 	inputs.forEach((input) => {
// 		if (input.placeholder.includes("Company"))
// 			data.companyName = input.value;
// 		if (input.placeholder.includes("Job Title"))
// 			data.jobTitle = input.value;
// 		if (input.placeholder.includes("job description"))
// 			data.jobDescription = input.value;
// 		if (input.placeholder.includes("Resume Highlights"))
// 			data.resumeHighlights = input.value;
// 	});

// 	console.log("Form data:", data);

// 	// Validate required fields
// 	if (!data.companyName || !data.jobTitle || !data.jobDescription) {
// 		alert("Please fill in all required fields.");
// 		return;
// 	}

// 	// Show progress
// 	showGenerationProgress();

// 	// Simulate API call (replace with actual API call)
// 	simulateGeneration(data);
// }

// function showGenerationProgress() {
// 	const progressContainer = document.getElementById("progress-container");
// 	const generateBtnText = document.getElementById("generate-btn-text");
// 	const generateBtnSpinner = document.getElementById("generate-btn-spinner");
// 	const progressFill = document.getElementById("progress-fill");
// 	const progressPercent = document.getElementById("progress-percent");

// 	if (progressContainer) progressContainer.classList.remove("hidden");
// 	if (generateBtnText) generateBtnText.textContent = "Generating...";
// 	if (generateBtnSpinner) generateBtnSpinner.classList.remove("hidden");

// 	// Animate progress
// 	let progress = 0;
// 	const interval = setInterval(() => {
// 		progress += Math.random() * 15;
// 		if (progress > 90) progress = 90;

// 		if (progressFill) progressFill.style.width = `${progress}%`;
// 		if (progressPercent)
// 			progressPercent.textContent = `${Math.round(progress)}%`;

// 		if (progress >= 90) {
// 			clearInterval(interval);
// 		}
// 	}, 200);
// }

// function simulateGeneration(data) {
// 	// Simulate API delay
// 	setTimeout(() => {
// 		const mockCoverLetter = generateMockCoverLetter(data);

// 		// Complete progress
// 		const progressFill = document.getElementById("progress-fill");
// 		const progressPercent = document.getElementById("progress-percent");
// 		if (progressFill) progressFill.style.width = "100%";
// 		if (progressPercent) progressPercent.textContent = "100%";

// 		// Hide progress after a moment
// 		setTimeout(() => {
// 			hideGenerationProgress();
// 			showCoverLetterPreview(mockCoverLetter);
// 			switchTab("preview");
// 		}, 500);
// 	}, 3000);
// }

// function hideGenerationProgress() {
// 	const progressContainer = document.getElementById("progress-container");
// 	const generateBtnText = document.getElementById("generate-btn-text");
// 	const generateBtnSpinner = document.getElementById("generate-btn-spinner");

// 	if (progressContainer) progressContainer.classList.add("hidden");
// 	if (generateBtnText) generateBtnText.textContent = "Generate Cover Letter";
// 	if (generateBtnSpinner) generateBtnSpinner.classList.add("hidden");
// }

// function generateMockCoverLetter(data) {
// 	return `Dear Hiring Manager,

// I am writing to express my strong interest in the ${
// 		data.jobTitle
// 	} position at ${
// 		data.companyName
// 	}. With my background in software development and passion for innovative technology solutions, I am excited about the opportunity to contribute to your team.

// In my previous roles, I have successfully delivered high-quality software solutions that align perfectly with the requirements outlined in your job description. My experience includes:

// • Developing scalable web applications using modern frameworks
// • Collaborating with cross-functional teams to deliver projects on time
// • Implementing best practices for code quality and testing
// • Contributing to open-source projects and staying current with industry trends

// ${
// 	data.resumeHighlights
// 		? `Key highlights from my experience include:\n${data.resumeHighlights}\n\n`
// 		: ""
// }I am particularly drawn to ${
// 		data.companyName
// 	} because of your commitment to innovation and excellence in the technology space. I believe my skills and enthusiasm would make me a valuable addition to your team.

// Thank you for considering my application. I look forward to the opportunity to discuss how I can contribute to ${
// 		data.companyName
// 	}'s continued success.

// Best regards,
// [Your Name]`;
// }

// function showCoverLetterPreview(coverLetter) {
// 	const previewElement = document.getElementById("cover-letter-preview");
// 	if (previewElement) {
// 		previewElement.innerHTML = `<div style="white-space: pre-wrap; line-height: 1.6;">${coverLetter}</div>`;
// 	}
// }

// // Utility functions
// function copyToClipboard() {
// 	const previewElement = document.getElementById("cover-letter-preview");
// 	if (previewElement) {
// 		const text = previewElement.textContent;
// 		navigator.clipboard.writeText(text).then(() => {
// 			// Show temporary success message
// 			const button = event.target;
// 			const originalText = button.innerHTML;
// 			button.innerHTML = "<span>✓</span> Copied!";
// 			button.style.background = "var(--success-color)";
// 			button.style.color = "white";

// 			setTimeout(() => {
// 				button.innerHTML = originalText;
// 				button.style.background = "";
// 				button.style.color = "";
// 			}, 2000);
// 		});
// 	}
// }

// function editCoverLetter() {
// 	switchTab("generate");
// }

// function saveCoverLetter() {
// 	// Implement save functionality
// 	alert("Cover letter saved to your library!");
// }

// // Global function for welcome state button
// // function handleSignIn() {
// //   const signinButton = document.getElementById("signin-btn");
// //   if (signinButton) {
// //     signinButton.click();
// //   }
// // }
