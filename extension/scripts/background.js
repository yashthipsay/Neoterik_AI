// Configuration - Update this with your actual API endpoint
const API_BASE_URL = "http://localhost:8000";
const EXTENSION_CALLBACK_URL = "http://localhost:3000/auth/extension-callback/";

// Store currently detected job pages to avoid redundant notifications
const detectedJobPages = new Set();

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
	if (request.action === "checkUrl") {
		const tabId = sender.tab?.id;
		if (!tabId) {
			console.error("No tab ID found in sender");
			sendResponse({ success: false, error: "No tab ID" });
			return false;
		}

		// Check the URL with the API
		checkUrlWithApi(request.url, tabId)
			.then((isJobPage) => {
				sendResponse({
					success: true,
					isJobPage: isJobPage,
				});
			})
			.catch((error) => {
				console.error("Error in URL check:", error);
				sendResponse({
					success: false,
					error: error.message,
				});
			});

		return true; // Indicates async response
	}

	// Handle request to extract job description
	else if (request.action === "extractJobDescription") {
		const tabId = sender.tab?.id;
		if (!tabId) {
			sendResponse({ success: false, error: "No tab ID" });
			return false;
		}

		// Add new handler for auth success from content script
		if (request.action === "authSuccess" && request.session) {
			console.log(
				"Background: Received auth success from content script:",
				message.session
			);

			// Store the session data
			chrome.storage.local.set(
				{
					isLoggedIn: true,
					user: request.session.user,
					authToken: request.session.token, // If you're passing a token
				},
				() => {
					// Notify popup about login status change
					notifyLoginStatusChanged();

					// Close the auth tab if we have its ID
					if (request.tabId) {
						setTimeout(() => {
							chrome.tabs
								.remove(request.tabId)
								.catch((err) =>
									console.warn(
										"Background: Failed to close callback tab:",
										err
									)
								);
						}, 2000);
					}
				}
			);

			sendResponse({ success: true });
			return true;
		}

		// Execute a content script to try to extract the job description
		chrome.scripting
			.executeScript({
				target: { tabId: tabId },
				function: () => {
					// This function runs in the context of the page
					function findJobDescription() {
						console.log(
							"Attempting to find 'Additional Information' or 'Cover Letter' sections..."
						);

						// Look for headings or labels containing the target phrases
						const potentialLabels = Array.from(
							document.querySelectorAll(
								"h1, h2, h3, h4, h5, h6, strong, b, label, dt, .form-label"
							)
						);
						let foundText = null;

						for (const labelElement of potentialLabels) {
							const labelText = labelElement.textContent.trim();

							if (
								/additional information|cover letter/i.test(
									labelText
								)
							) {
								console.log(
									"Found potential label:",
									labelText,
									labelElement
								);

								// Try to get text from the next sibling element
								let nextElement =
									labelElement.nextElementSibling;
								if (
									nextElement &&
									nextElement.textContent.trim().length > 50
								) {
									console.log(
										"Found text in next sibling:",
										nextElement.textContent.trim()
									);
									foundText = nextElement.textContent.trim();
									break; // Stop searching once found
								}

								// If next sibling didn't work, try the parent's next sibling (common in definition lists dt/dd)
								if (!foundText && labelElement.parentElement) {
									nextElement =
										labelElement.parentElement
											.nextElementSibling;
									if (
										nextElement &&
										nextElement.textContent.trim().length >
											50
									) {
										console.log(
											"Found text in parent's next sibling:",
											nextElement.textContent.trim()
										);
										foundText =
											nextElement.textContent.trim();
										break;
									}
								}

								// If still not found, try finding a nearby textarea or content div
								// This is more complex and might need refinement based on common structures
								let parent = labelElement.parentElement;
								let attempts = 0;
								while (parent && attempts < 3) {
									const nearbyTextarea =
										parent.querySelector("textarea");
									if (
										nearbyTextarea &&
										nearbyTextarea.value.trim().length > 50
									) {
										console.log(
											"Found text in nearby textarea:",
											nearbyTextarea.value.trim()
										);
										foundText = nearbyTextarea.value.trim();
										break;
									}
									const nearbyDiv = parent.querySelector(
										'div[class*="content"], div[class*="description"], div.ProseMirror'
									); // Common rich text editor class
									if (
										nearbyDiv &&
										nearbyDiv.textContent.trim().length > 50
									) {
										console.log(
											"Found text in nearby content div:",
											nearbyDiv.textContent.trim()
										);
										foundText =
											nearbyDiv.textContent.trim();
										break;
									}
									parent = parent.parentElement;
									attempts++;
								}
								if (foundText) break;
							}
						}

						if (foundText) {
							return foundText;
						}

						console.log(
							"Specific sections not found, falling back to body text."
						);
						// Fallback: return the body text if specific sections aren't found
						return document.body.innerText.substring(0, 5000);
					}

					return findJobDescription();
				},
			})
			.then((results) => {
				if (results && results[0] && results[0].result) {
					sendResponse({
						success: true,
						description: results[0].result,
					});
				} else {
					sendResponse({
						success: false,
						error: "No job description found",
					});
				}
			})
			.catch((error) => {
				console.error("Error extracting job description:", error);
				sendResponse({
					success: false,
					error: error.message,
				});
			});

		return true; // Indicates async response
	}
});

// Check URL when tab is updated
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
	// Log all status changes for debugging
	// console.log(`Tab ${tabId} updated: Status=${changeInfo.status}, URL=${tab.url}`);

	// --- Handle Auth Callback ---
	// Use 'complete' status and check URL prefix
	if (
		changeInfo.status === "complete" &&
		tab.url.includes("/auth/extension-callback/")
	) {
		handleAuthCallback(tabId, tab.url);
		return; // Don't process further if it's the callback URL
	}

	// --- Handle Job Page Check ---
	// Only check if the page has finished loading, has a valid HTTP/S URL,
	// is not an extension page, and is not the callback URL.
	if (
		changeInfo.status === "complete" &&
		tab.url &&
		tab.url.includes("/auth/extension-callback")
	) {
		// Inject a content script to listen for the postMessage
		chrome.scripting
			.executeScript({
				target: { tabId },
				function: function () {
					// This runs in the page context
					window.addEventListener(
						"message",
						function (event) {
							// In production, check event.origin
							if (
								event.data &&
								event.data.type === "EXTENSION_AUTH_SUCCESS"
							) {
								// Forward the message to the background script
								chrome.runtime.sendMessage({
									action: "authSuccess",
									session: event.data.session,
									tabId: chrome.devtools
										? null
										: chrome.devtools.inspectedWindow.tabId,
								});
							}
						},
						false
					);

					// Also set up a MutationObserver to detect title changes
					const observer = new MutationObserver((mutations) => {
						if (document.title === "AUTH_SUCCESS") {
							chrome.runtime.sendMessage({
								action: "authTitleDetected",
								tabId: chrome.devtools
									? null
									: chrome.devtools.inspectedWindow.tabId,
							});
						}
					});

					observer.observe(document.querySelector("title"), {
						subtree: true,
						characterData: true,
						childList: true,
					});
				},
			})
			.catch((err) =>
				console.error("Error injecting auth listener script:", err)
			);
	}
});

// Clear badge when tab is removed
chrome.tabs.onRemoved.addListener((tabId) => {
	chrome.action.setBadgeText({ text: "", tabId: tabId });
});

// Function to check if URL is a job application page via the API
async function checkUrlWithApi(url, tabId) {
	try {
		console.log("Checking URL with API:", url);

		const response = await fetch(`${API_BASE_URL}/check-url`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ url }),
		});

		const data = await response.json();
		console.log("🔍 Graph API Response:", data);

		if (data.is_job_application && data.parsed_output) {
			            console.log("[checkUrlWithApi] ✅ check-url endpoint succeeded for:", url);
			// ✅ Store parsed job info for popup or auto-fill
			chrome.storage.local.set({
				currentJobPage: {
					url: url,
					detected: Date.now(),
					jobData: data.parsed_output,
				},
			});

			// ✅ Show job badge
			chrome.action.setBadgeText({ text: "JOB", tabId: tabId });
			chrome.action.setBadgeBackgroundColor({
				color: "#419D78",
				tabId: tabId,
			});

			return true;
		} else {
console.log("[checkUrlWithApi] ❌ check-url endpoint did not detect a job page for:", url);
			// ❌ Not a job page (or failed), clear badge
			chrome.action.setBadgeText({ text: "", tabId: tabId });
			return false;
		}
	} catch (error) {
		console.error("❌ API fallback error:", error);

		// 🔁 Fallback to legacy job pattern detection
console.log("[checkUrlWithApi] ⚠️ Falling back to legacy job board pattern detection for:", url);
		const KNOWN_JOB_BOARD_PATTERNS = [
			/job-boards\.greenhouse\.io\/.+\/jobs\/\d+/,
			/jobs\.lever\.co\/.+\/\d+/,
			/boards\.greenhouse\.io\/.+\/#application_form/,
			/apply\.workable\.com\/.+\/j\/.+/,
			/jobs\.ashbyhq\.com\/.+\/.+/,
			/careers\.smartrecruiters\.com\/.+\/job\/.+/,
			/workday\..+\/careers\/.+\/job\/.+/,
			/linkedin\.com\/jobs\/view\/.+/,
			/indeed\.com\/.+\/viewjob/,
			/wellfound\.com\/jobs\/.+/,
		];

		const isLegacyMatch = KNOWN_JOB_BOARD_PATTERNS.some((p) => p.test(url));

		if (isLegacyMatch) {
	console.log("[checkUrlWithApi] 🟡 Legacy pattern matched for:", url);
			chrome.action.setBadgeText({ text: "JOB", tabId: tabId });
			chrome.action.setBadgeBackgroundColor({
				color: "#777",
				tabId: tabId,
			});

			chrome.storage.local.set({
				currentJobPage: {
					url: url,
					detected: Date.now(),
					jobData: null, // Legacy match, no graph data
				},
			});

			return true;
		}
		console.log("[checkUrlWithApi] ⛔️ No legacy pattern match for:", url);
        return false;
	}
}

// async function checkUrlWithApi(url, tabId) {
// 	try {
// 		console.log("Checking URL with API:", url);

// 		const response = await fetch(`${API_BASE_URL}/check-url`, {
// 			method: "POST",
// 			headers: {
// 				"Content-Type": "application/json",
// 			},
// 			body: JSON.stringify({ url }),
// 		});

// 		if (!response.ok) {
// 			throw new Error(
// 				`API request failed with status ${response.status}`
// 			);
// 		}

// 		const data = await response.json();
// 		console.log("API Response:", data);

// 		// If it's a job application page, update the extension UI
// 		if (data.is_job_application) {
// 			// Set badge on the extension icon
// 			chrome.action.setBadgeText({
// 				text: "JOB",
// 				tabId: tabId,
// 			});
// 			chrome.action.setBadgeBackgroundColor({
// 				color: "#419D78",
// 				tabId: tabId,
// 			});

// 			// Store this URL as detected
// 			detectedJobPages.add(url);

// 			// Store job URL in local storage for the popup
// 			chrome.storage.local.set({
// 				currentJobPage: {
// 					url: url,
// 					detected: Date.now(),
// 				},
// 			});

// 			return true;
// 		} else {
// 			// Clear badge if not a job page
// 			chrome.action.setBadgeText({ text: "", tabId: tabId });
// 			return false;
// 		}
// 	} catch (error) {
// 		console.error("Error checking URL with API:", error);
// 		return false;
// 	}
// }

// --- Function to handle sending login status change message ---
function notifyLoginStatusChanged() {
	chrome.runtime
		.sendMessage({ action: "loginStatusChanged" })
		.catch((error) => {
			if (error.message.includes("Could not establish connection")) {
				console.log(
					"Background: Popup not open, message not sent (this is expected)."
				);
			} else {
				console.error(
					"Background: Error sending login status message:",
					error
				);
			}
		});
}

// Track which tabs we've already handled for auth
const handledAuthTabs = new Set();

// --- Function to process the authentication callback ---
async function handleAuthCallback(tabId, url) {
	if (handledAuthTabs.has(tabId)) {
		// Already handled, don't process again
		return;
	}
	handledAuthTabs.add(tabId);

	console.log(
		"Background: Handling auth callback for tab",
		tabId,
		"URL:",
		url
	);

	try {
		// Fetch the actual session data
		const response = await fetch("http://localhost:3000/api/auth/session", {
			method: "GET",
			headers: { Accept: "application/json" },
			credentials: "include",
		});

		if (response.ok) {
			const session = await response.json();
			console.log("Background: Session data received:", session);

			if (session && session.user) {
				await chrome.storage.local.set({
					isLoggedIn: true,
					user: session.user,
				});

				notifyLoginStatusChanged();

				// Close the tab after a short delay
				setTimeout(() => {
					chrome.tabs.remove(tabId).catch((err) => {
						console.warn(
							"Background: Failed to close callback tab:",
							err
						);
					});
				}, 1000);

				return;
			}
		}

		console.error("Background: Failed to fetch or parse session");
	} catch (error) {
		console.error("Background: Error during auth callback:", error);
	}
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
	if (request.action === "checkUrl") {
		const tabId = sender.tab?.id;
		if (!tabId) {
			console.error("No tab ID found in sender");
			sendResponse({ success: false, error: "No tab ID" });
			return false;
		}

		// Check the URL with the API
		checkUrlWithApi(request.url, tabId)
			.then((isJobPage) => {
				sendResponse({
					success: true,
					isJobPage: isJobPage,
				});
			})
			.catch((error) => {
				console.error("Error in URL check:", error);
				sendResponse({
					success: false,
					error: error.message,
				});
			});

		return true; // Indicates async response
	}

	// Handle request to extract job description
	else if (request.action === "extractJobDescription") {
		const tabId = sender.tab?.id;
		if (!tabId) {
			sendResponse({ success: false, error: "No tab ID" });
			return false;
		}

		// Execute a content script to try to extract the job description
		chrome.scripting
			.executeScript({
				target: { tabId: tabId },
				function: () => {
					// This function runs in the context of the page
					function findJobDescription() {
						console.log(
							"Attempting to find 'Additional Information' or 'Cover Letter' sections..."
						);

						// Look for headings or labels containing the target phrases
						const potentialLabels = Array.from(
							document.querySelectorAll(
								"h1, h2, h3, h4, h5, h6, strong, b, label, dt, .form-label"
							)
						);
						let foundText = null;

						for (const labelElement of potentialLabels) {
							const labelText = labelElement.textContent.trim();

							if (
								/additional information|cover letter/i.test(
									labelText
								)
							) {
								console.log(
									"Found potential label:",
									labelText,
									labelElement
								);

								// Try to get text from the next sibling element
								let nextElement =
									labelElement.nextElementSibling;
								if (
									nextElement &&
									nextElement.textContent.trim().length > 50
								) {
									console.log(
										"Found text in next sibling:",
										nextElement.textContent.trim()
									);
									foundText = nextElement.textContent.trim();
									break; // Stop searching once found
								}

								// If next sibling didn't work, try the parent's next sibling (common in definition lists dt/dd)
								if (!foundText && labelElement.parentElement) {
									nextElement =
										labelElement.parentElement
											.nextElementSibling;
									if (
										nextElement &&
										nextElement.textContent.trim().length >
											50
									) {
										console.log(
											"Found text in parent's next sibling:",
											nextElement.textContent.trim()
										);
										foundText =
											nextElement.textContent.trim();
										break;
									}
								}

								// If still not found, try finding a nearby textarea or content div
								// This is more complex and might need refinement based on common structures
								let parent = labelElement.parentElement;
								let attempts = 0;
								while (parent && attempts < 3) {
									const nearbyTextarea =
										parent.querySelector("textarea");
									if (
										nearbyTextarea &&
										nearbyTextarea.value.trim().length > 50
									) {
										console.log(
											"Found text in nearby textarea:",
											nearbyTextarea.value.trim()
										);
										foundText = nearbyTextarea.value.trim();
										break;
									}
									const nearbyDiv = parent.querySelector(
										'div[class*="content"], div[class*="description"], div.ProseMirror'
									); // Common rich text editor class
									if (
										nearbyDiv &&
										nearbyDiv.textContent.trim().length > 50
									) {
										console.log(
											"Found text in nearby content div:",
											nearbyDiv.textContent.trim()
										);
										foundText =
											nearbyDiv.textContent.trim();
										break;
									}
									parent = parent.parentElement;
									attempts++;
								}
								if (foundText) break;
							}
						}

						if (foundText) {
							return foundText;
						}

						console.log(
							"Specific sections not found, falling back to body text."
						);
						// Fallback: return the body text if specific sections aren't found
						return document.body.innerText.substring(0, 5000);
					}

					return findJobDescription();
				},
			})
			.then((results) => {
				if (results && results[0] && results[0].result) {
					sendResponse({
						success: true,
						description: results[0].result,
					});
				} else {
					sendResponse({
						success: false,
						error: "No job description found",
					});
				}
			})
			.catch((error) => {
				console.error("Error extracting job description:", error);
				sendResponse({
					success: false,
					error: error.message,
				});
			});

		return true; // Indicates async response
	}
});

// Check URL when tab is updated
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
	// Log all status changes for debugging
	// console.log(`Tab ${tabId} updated: Status=${changeInfo.status}, URL=${tab.url}`);

	// --- Handle Auth Callback ---
	// Use 'complete' status and check URL prefix
	if (
		changeInfo.status === "complete" &&
		tab.url &&
		tab.url.includes("/auth/extension-callback/")
	) {
		handleAuthCallback(tabId, tab.url);
		return;
	}

	// --- Handle Job Page Check ---
	// Only check if the page has finished loading, has a valid HTTP/S URL,
	// is not an extension page, and is not the callback URL.
	if (
		changeInfo.status === "complete" &&
		tab.url &&
		tab.url.startsWith("http") && // Covers http and https
		!tab.url.startsWith("chrome-extension://") &&
		!tab.url.startsWith(EXTENSION_CALLBACK_URL)
	) {
		// console.log(`Checking job status for completed tab ${tabId}: ${tab.url}`); // Optional logging
		checkUrlWithApi(tab.url, tabId);
	}
});

// Clear badge when tab is removed
chrome.tabs.onRemoved.addListener((tabId) => {
	chrome.action.setBadgeText({ text: "", tabId: tabId });
});
