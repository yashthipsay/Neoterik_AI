console.log("Neoterik Cover Letter Assistant: Content script loaded");

// Debounce helper
function debounce(func, wait) {
	let timeout;
	return function (...args) {
		clearTimeout(timeout);
		timeout = setTimeout(() => func.apply(this, args), wait);
	};
}

// Debounced job page check
const checkCurrentUrlDebounced = debounce(function () {
	const url = window.location.href;

	chrome.runtime.sendMessage({ action: "checkUrl", url }, (response) => {
		if (chrome.runtime.lastError) {
			console.error("Runtime error:", chrome.runtime.lastError);
			return;
		}

		if (response?.success && response.isJobPage) {
			console.log("✅ Job page detected!");

			// Inject banner only once
			if (!document.getElementById("neoterik-job-detected")) {
				chrome.runtime.sendMessage(
					{ action: "shouldInjectBanner", url },
					(response) => {
						if (response?.allow) {
							injectJobPageNotification();
						}
					}
				);
			}
		}
	});
}, 1000);

// Add a global flag
let isAgentRunning = false;

// Inject the banner
function injectJobPageNotification() {
	let clicked = false;
	const notification = document.createElement("div");
	notification.id = "neoterik-job-detected";
	notification.style.cssText = `
		position: fixed;
		top: 16px;
		right: 16px;
		background-color: #419D78;
		color: white;
		padding: 12px 16px;
		border-radius: 8px;
		font-family: 'Segoe UI', Tahoma, sans-serif;
		font-size: 14px;
		z-index: 9999;
		box-shadow: 0 4px 12px rgba(0,0,0,0.15);
		display: flex;
		align-items: center;
		cursor: pointer;
		max-width: 300px;
	`;

	notification.innerHTML = `
		<div style="margin-right: 12px;">
			<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
				<path d="M21 7V17C21 18.1046 20.1046 19 19 19H5C3.89543 19 3 18.1046 3 17V7M21 7C21 5.89543 20.1046 5 19 5H5C3.89543 5 3 5.89543 3 7M21 7L12 13L3 7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
			</svg>
		</div>
		<div>
			<div style="font-weight: 600; margin-bottom: 4px;">Neoterik detected a job!</div>
			<div style="font-size: 12px;">Click to generate a cover letter</div>
		</div>
	`;

	// Run agent on click
	notification.addEventListener("click", () => {
		if (clicked || isAgentRunning) return; // Prevent multiple clicks or if agent running
		clicked = true;
		isAgentRunning = true;
		console.log("📧 Banner clicked: Starting job agent...");
		notification.style.pointerEvents = "none";
		notification.style.opacity = "0.5";

		// Add spinner
		const spinner = document.createElement("div");
		spinner.innerHTML = `<div class="loader">⏳</div>`;
		spinner.style.marginLeft = "12px";
		notification.appendChild(spinner);

		chrome.runtime.sendMessage({ action: "run_job_agent" }, () => {
			isAgentRunning = false;
		});

		// Remove banner
		setTimeout(() => notification.remove(), 500);

		// Open popup after delay
		setTimeout(() => {
			chrome.runtime.sendMessage({ action: "openPopup" });
		}, 2000);
	});

	// Add dismiss (×) button
	const dismissBtn = document.createElement("button");
	dismissBtn.style.cssText = `
		background: transparent;
		border: none;
		color: white;
		font-size: 16px;
		cursor: pointer;
		margin-left: auto;
	`;
	dismissBtn.innerHTML = "×";
	dismissBtn.addEventListener("click", (e) => {
		e.stopPropagation();
		notification.remove();
	});
	notification.appendChild(dismissBtn);

	document.body.appendChild(notification);

	// Auto-remove after 10s
	// setTimeout(() => {
	// 	if (notification.parentNode) {
	// 		notification.remove();
	// 	}
	// }, 10000);
}

// Run on initial load
setTimeout(() => {
	checkCurrentUrlDebounced();
}, 2000);

// Watch for URL changes (SPA, client routing)
let lastUrl = location.href;
new MutationObserver(() => {
	const url = location.href;
	if (url !== lastUrl) {
		lastUrl = url;
		checkCurrentUrlDebounced();
	}
}).observe(document, { subtree: true, childList: true });

// For history navigation
window.addEventListener("popstate", checkCurrentUrlDebounced);
