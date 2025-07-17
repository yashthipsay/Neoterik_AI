// === Configuration ===

const API_BASE_URL = "http://localhost:8000";
const EXTENSION_CALLBACK_URL = "http://localhost:3000/auth/extension-callback/";

// Used for debouncing URL checks per tab
const checkUrlTimers = {};
const detectedJobsPerTab = {};

// === Utility: Notify popup of login ===
function notifyLoginStatusChanged() {
	chrome.runtime
		.sendMessage({ action: "loginStatusChanged" })
		.catch((error) => {
			if (!error.message.includes("Could not establish connection")) {
				console.error(
					"Background: Error sending login status message:",
					error
				);
			}
		});
}

// === Sign Out Handler ===
function handleSignOut() {
	console.log("[Extension] Initiating sign out...");
	// Open signout in a new tab to clear NextAuth cookies
	chrome.tabs.create(
		{ url: "http://localhost:3000/api/auth/signout?callbackUrl=/" },
		(tab) => {
			// Wait a moment, then clear extension storage and reload home page
			setTimeout(() => {
				chrome.storage.local.clear(() => {
					notifyLoginStatusChanged();
					chrome.tabs.update(tab.id, {
						url: "http://localhost:3000",
						active: true,
					});
					console.log(
						"[Extension] Sign out complete, storage cleared, home page opened."
					);
				});
			}, 1500); // Give NextAuth time to clear cookies
		}
	);
}

// === Auth Callback Handler ===
const handledAuthTabs = new Set();
async function handleAuthCallback(tabId, url) {
	if (handledAuthTabs.has(tabId)) return;
	handledAuthTabs.add(tabId);

	try {
		const response = await fetch("http://localhost:3000/api/auth/session", {
			method: "GET",
			headers: { Accept: "application/json" },
			credentials: "include",
		});

		if (response.ok) {
			const session = await response.json();
			if (session && session.user) {
				await chrome.storage.local.set({
					isLoggedIn: true,
					user: session.user,
					userId: session.user.id,
				});
				console.log("🔑 Session result from server:", session);
				notifyLoginStatusChanged();
				setTimeout(() => {
					chrome.tabs.remove(tabId).catch((err) => {
						if (
							err &&
							err.message &&
							err.message.includes("No tab with id")
						) {
							// Silently ignore this error
							return;
						}
						// Log other errors
						console.warn("chrome.tabs.remove error:", err);
					});
				}, 1000);
			}
		} else {
			console.error("Auth callback failed to fetch session");
		}
	} catch (error) {
		console.error("Error during auth callback:", error);
	}
}

function shouldProceedWithDetection(url, callback) {
	chrome.storage.local.get("jobSession", (data) => {
		const session = data.jobSession;
		if (session?.isLocked && !session?.isCoverLetterGenerated) {
			console.warn("🔒 Detection blocked due to locked session.");
			return callback(false);
		}
		callback(true);
	});
}

// === Core Job Page Detection ===
// Track cleanup timeouts per session
let sessionCleanupTimeout = null;

async function checkUrlWithApi(url, tabId) {
	try {
		const response = await fetch(`${API_BASE_URL}/check-url`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ url }),
		});

		if (!response.ok) throw new Error(`Status ${response.status}`);

		const data = await response.json();

		if (data.is_job_application) {
			detectedJobsPerTab[tabId] = {
				url,
				jobData: data?.parsed_output || null,
				detectedAt: Date.now(),
			};
			chrome.action.setBadgeText({ text: "JOB", tabId });
			chrome.action.setBadgeBackgroundColor({ color: "#419D78", tabId });

			await chrome.storage.local.set({
				jobSession: {
					jobUrl: url,
					isJobDetected: true,
					isAgentInProgress: false,
					isAgentFinished: false,
					isLocked: false,
					isCoverLetterGenerated: false,
					isCoverLetterGenerating: false,
					coverLetterError: null,
					isUserConfirmed: null,
					timestamp: Date.now(),
				},
				currentJobPage: {
					url,
					detected: Date.now(),
					detectedFrom: "check-url",
					jobData: data?.parsed_output || null,
				},
			});
			// Clear any previous cleanup timeout if new job detected
			if (sessionCleanupTimeout) {
				clearTimeout(sessionCleanupTimeout);
				sessionCleanupTimeout = null;
			}

			return true;
		} else {
			chrome.action.setBadgeText({ text: "", tabId });
			return false;
		}
	} catch (error) {
		console.error("Error checking URL with API:", error);
		return false;
	}
}

async function handleGenerateCoverLetter(data) {
	// Set flag in storage
	const { jobSession, currentJobPage } = await chrome.storage.local.get([
		"jobSession",
		"currentJobPage",
		"userId",
	]);
	await chrome.storage.local.set({
		jobSession: {
			...jobSession,
			isLocked: true, // lock session while generating
			isCoverLetterGenerating: true,
			isCoverLetterGenerated: false,
			coverLetterError: null,
		},
	});
	try {
		const payload = { user_id: userId, ...data };
		const res = await fetch(`${API_BASE_URL}/generate-cover-letter`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});
		const result = await res.json();
		if (res.ok && result.cover_letter) {
			await chrome.storage.local.set({
				jobSession: {
					jobUrl: jobSession.jobUrl,
					isJobDetected: false,
					isAgentInProgress: false,
					isAgentFinished: false,
					isLocked: false,
					isCoverLetterGenerated: true,
					isCoverLetterGenerating: false,
					isUserConfirmed: null,
					coverLetter: result.cover_letter,
					coverLetterError: null,
					timestamp: Date.now(),
				},
				currentJobPage: {
					...(currentJobPage || {}),
					coverLetter: result.cover_letter,
				},
			});
			chrome.runtime.sendMessage({
				action: "coverLetterGenerated",
				coverLetter: result.cover_letter,
			});
		} else {
			await chrome.storage.local.set({
				jobSession: {
					...jobSession,
					isLocked: false,
					isCoverLetterGenerating: false,
					isCoverLetterGenerated: false,
					coverLetterError:
						result.error || "Failed to generate cover letter",
				},
			});
			chrome.runtime.sendMessage({
				action: "coverLetterError",
				error: result.error || "Failed to generate cover letter",
			});
		}
	} catch (err) {
		await chrome.storage.local.set({
			jobSession: {
				...jobSession,
				isLocked: false,
				isCoverLetterGenerating: false,
				isCoverLetterGenerated: false,
				coverLetterError: err.message || "Network error",
			},
		});
		chrome.runtime.sendMessage({
			action: "coverLetterError",
			error: err.message || "Network error",
		});
	}
}

// Add a global flag for agent running
let isAgentRunning = false;

// === Content Script Communication ===
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
	if (request.action === "generateCoverLetter") {
		console.log("[Background] Received generateCoverLetter message");
		handleGenerateCoverLetter(request.data);
		return true;
	}

	if (request.action === "removeBanner") {
		const banner = document.getElementById("neoterik-job-detected");
		if (banner) banner.remove();
	}

	const tabId = sender.tab?.id;

	if (!tabId) {
		sendResponse({ success: false, error: "No tab ID" });
		return false;
	}

	// === Listen for signOut action ===
	if (request.action === "signOut") {
		handleSignOut();
		sendResponse({ success: true });
		return true;
	}

	if (request.action === "checkUrl") {
		shouldProceedWithDetection(request.url, (canProceed) => {
			if (!canProceed)
				return sendResponse({ success: false, blocked: true });

			checkUrlWithApi(request.url, tabId)
				.then((isJobPage) => sendResponse({ success: true, isJobPage }))
				.catch((error) =>
					sendResponse({ success: false, error: error.message })
				);
		});
		return true;
	}

	if (request.action === "shouldInjectBanner") {
		shouldProceedWithDetection(request.url, (canProceed) => {
			sendResponse({ allow: canProceed });
		});
		return true;
	}

	if (request.action === "run_job_agent") {
		if (isAgentRunning) {
			sendResponse({ success: false, error: "Agent already running" });
			return true;
		}
		isAgentRunning = true;
		chrome.storage.local.get(
			["jobSession", "currentJobPage"],
			async (data) => {
				let session = data.jobSession;
				// If session is missing or jobUrl is missing, try to recover from currentJobPage
				if (!session || !session.jobUrl) {
					if (data.currentJobPage && data.currentJobPage.url) {
						// Recreate jobSession from currentJobPage
						session = {
							jobUrl: data.currentJobPage.url,
							isJobDetected: true,
							isAgentInProgress: false,
							isAgentFinished: false,
							isLocked: false,
							isCoverLetterGenerated: false,
							isUserConfirmed: null,
							timestamp: Date.now(),
						};
						await chrome.storage.local.set({ jobSession: session });
						console.warn(
							"[Background] Recovered jobSession from currentJobPage."
						);
					} else {
						console.error(
							"[Background] No jobSession or currentJobPage found. Cannot start agent."
						);
						chrome.runtime.sendMessage({
							action: "agentError",
							error: "No job detected. Please refresh the page and try again.",
						});
						isAgentRunning = false;
						return;
					}
				}

				// Set progress state
				await chrome.storage.local.set({
					jobSession: {
						...session,
						isAgentInProgress: true,
						isAgentFinished: false,
					},
				});
				console.log("[Background] Starting agent for:", session.jobUrl);

				try {
					// Log the URL being sent to the agent
					console.log(
						"[Background] Calling /run-agent with URL:",
						session.jobUrl
					);
					const response = await fetch(`${API_BASE_URL}/run-agent`, {
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({ url: session.jobUrl }),
					});
					const result = await response.json();
					// Log the result
					console.log("[Background] Agent API response:", result);

					// Validate result: must be a non-empty object with at least job_title or company_name
					const isValidResult =
						result &&
						(result.job_title ||
							result.company_name ||
							Object.keys(result).length > 0);
					if (response.ok && isValidResult) {
						await chrome.storage.local.set({
							currentJobPage: {
								url: session.jobUrl,
								detected: Date.now(),
								jobData: result,
							},
							jobSession: {
								...session,
								isAgentInProgress: false,
								isAgentFinished: true,
								isLocked: true,
							},
						});
						// Remind to complete job after 5 minutes if not completed
						if (sessionCleanupTimeout) {
							clearTimeout(sessionCleanupTimeout);
							sessionCleanupTimeout = null;
						}
						sessionCleanupTimeout = setTimeout(async () => {
							const { jobSession } =
								await chrome.storage.local.get("jobSession");
							if (
								jobSession?.isLocked &&
								!jobSession?.isCoverLetterGenerated
							) {
								chrome.notifications.create({
									type: "basic",
									iconUrl: "icons/icon128.png",
									title: "⏳ You left a job incomplete",
									message:
										"You started generating a cover letter. Please complete or skip.",
									priority: 2,
								});
								// After notification, clear session/job data and unlock detection
								await chrome.storage.local.remove([
									"jobSession",
									"currentJobPage",
								]);
							}
							sessionCleanupTimeout = null;
						}, 5 * 60 * 1000); // 5 minutes
						chrome.runtime.sendMessage({ action: "agentFinished" });
						isAgentRunning = false;
					} else {
						// If result is invalid, log and show error
						console.error(
							"[Background] Agent returned invalid/empty result:",
							result
						);
						await chrome.storage.local.set({
							jobSession: {
								...session,
								isAgentInProgress: false,
								isAgentFinished: false,
								isLocked: false,
								agentError:
									result?.error ||
									"Agent returned no data. Please try again or check the job page.",
							},
						});
						chrome.runtime.sendMessage({
							action: "agentError",
							error:
								result?.error ||
								"Agent returned no data. Please try again or check the job page.",
						});
						isAgentRunning = false;
						return;
					}
				} catch (err) {
					await chrome.storage.local.set({
						jobSession: {
							...session,
							isLocked: false,
							isAgentInProgress: false,
							isAgentFinished: false,
							agentError: err.message || "Agent error",
						},
					});
					console.error("[Background] Agent network/error:", err);
					isAgentRunning = false;
				}
			}
		);
		return true;
	}

	if (request.action === "extractJobDescription") {
		chrome.scripting
			.executeScript({
				target: { tabId },
				function: () => {
					function findJobDescription() {
						const potentialLabels = Array.from(
							document.querySelectorAll(
								"h1, h2, h3, h4, h5, h6, strong, b, label, dt, .form-label"
							)
						);
						for (const label of potentialLabels) {
							const text = label.textContent.trim();
							if (
								/additional information|cover letter/i.test(
									text
								)
							) {
								let next =
									label.nextElementSibling ||
									label.parentElement?.nextElementSibling;
								if (next && next.textContent.trim().length > 50)
									return next.textContent.trim();

								let parent = label.parentElement;
								for (
									let i = 0;
									i < 3 && parent;
									i++, parent = parent.parentElement
								) {
									const textarea =
										parent.querySelector("textarea");
									if (textarea?.value.trim().length > 50)
										return textarea.value.trim();
									const div = parent.querySelector(
										"div[class*='content'], div[class*='description'], div.ProseMirror"
									);
									if (div?.textContent.trim().length > 50)
										return div.textContent.trim();
								}
							}
						}
						return document.body.innerText.substring(0, 5000);
					}
					return findJobDescription();
				},
			})
			.then((results) => {
				sendResponse({
					success: true,
					description: results?.[0]?.result || null,
				});
			})
			.catch((err) => {
				sendResponse({ success: false, error: err.message });
			});
		return true;
	}

	if (request.action === "authSuccess" && request.session) {
		chrome.storage.local.set(
			{
				isLoggedIn: true,
				user: request.session.user,
				authToken: request.session.token,
			},
			() => {
				chrome.tabs.query(
					{ url: "http://localhost:3000/*" },
					(tabs) => {
						if (tabs.length > 0) {
							chrome.tabs.update(tabs[0].id, { active: true });
						} else {
							chrome.tabs.create({
								url: "http://localhost:3000",
							});
						}
					}
				);
				notifyLoginStatusChanged();
				if (request.tabId) {
					setTimeout(() => {
						chrome.tabs.remove(request.tabId).catch((err) => {
							if (
								err &&
								err.message &&
								err.message.includes("No tab with id")
							) {
								// Silently ignore this error
								return;
							}
							// Log other errors
							console.warn("chrome.tabs.remove error:", err);
						});
					}, 2000);
				}
			}
		);
		sendResponse({ success: true });
		return true;
	}
});

// === Tab Updates ===
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
	if (changeInfo.status !== "complete" || !tab.url) return;

	if (tab.url.includes("/auth/extension-callback/")) {
		handleAuthCallback(tabId, tab.url);
		return;
	}

	if (
		tab.url.startsWith("http") &&
		!tab.url.startsWith("chrome-extension://") &&
		!tab.url.startsWith(EXTENSION_CALLBACK_URL)
	)
		if (checkUrlTimers[tabId]) clearTimeout(checkUrlTimers[tabId]);
	checkUrlTimers[tabId] = setTimeout(() => {
		shouldProceedWithDetection(tab.url, (canProceed) => {
			if (canProceed) checkUrlWithApi(tab.url, tabId);
		});
	}, 1500); // Add 1.5s debounce
});

chrome.tabs.onActivated.addListener(({ tabId }) => {
	//check if this tab has a detected job
	const jobInfo = detectedJobsPerTab[tabId];
	if (jobInfo) {
		// Ask content script to iject the banner
		chrome.tabs.sendMessage(tabId, { action: "injectBanner" });
	}
});

// === Tab Removed Cleanup ===
chrome.tabs.onRemoved.addListener((tabId) => {
	delete detectedJobsPerTab[tabId];
	chrome.action.setBadgeText({ text: "", tabId });
	if (checkUrlTimers[tabId]) {
		clearTimeout(checkUrlTimers[tabId]);
		delete checkUrlTimers[tabId];
	}
});

// function isPopupOpen(callback) {
// 	chrome.extension.getViews({ type: "popup" }).length > 0
// 		? callback(true)
// 		: callback(false);
// }
