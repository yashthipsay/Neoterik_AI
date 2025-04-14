// filepath: /home/wsl/ai_coverLetter/extension/scripts/popup.js
document.addEventListener('DOMContentLoaded', function() {
    // Check if we have a stored job description
    chrome.storage.local.get(['jobDescription', 'currentJobPage'], function(data) {
      if (data.jobDescription) {
        document.getElementById('job-description').value = data.jobDescription;
        // Clear it so it's not used again unless updated
        chrome.storage.local.remove('jobDescription');
      }
      
      // Add event listener to the "Generate Cover Letter" button
      const generateButton = document.querySelector('.button:not(.accent)');
      if (generateButton) {
        generateButton.addEventListener('click', function() {
          const jobDescription = document.getElementById('job-description').value;
        //   const resumeHighlights = document.getElementById('resume-highlights').value;
          
          if (!jobDescription) {
            alert('Please enter a job description.');
            return;
          }
          
          // TODO: Send to your API to generate cover letter
          // For now, just show a placeholder
          alert('Generating cover letter... This feature will be implemented soon!');
        });
      }
    });
  });