/*
 * JavaScript for Asaf Dabush's portfolio site
 *
 * This script populates the current year in the footer and pulls a handful of the
 * most recently updated public GitHub repositories from Asaf's account using the
 * GitHub REST API. The fetched data is inserted into the Projects section,
 * providing an at‑a‑glance view of ongoing work without the need to manually
 * update the page. If the request fails (due to rate limiting or offline
 * conditions) the section remains hidden.
 */

// Insert the current year into the footer
document.getElementById('year').textContent = new Date().getFullYear().toString();

// Function to fetch and render recent GitHub repositories
function loadGitHubProjects() {
  const username = 'asafdabush';
  const apiUrl = `https://api.github.com/users/${username}/repos?sort=updated&per_page=4`;

  fetch(apiUrl)
    .then((response) => {
      if (!response.ok) {
        throw new Error('GitHub API request failed');
      }
      return response.json();
    })
    .then((repos) => {
      const projectsContainer = document.getElementById('github-projects');
      // If no repos found, hide the subheading
      if (!repos || repos.length === 0) {
        return;
      }
      repos.forEach((repo) => {
        const card = document.createElement('article');
        card.className = 'project-card';
        // Repo name
        const title = document.createElement('h3');
        title.textContent = repo.name;
        // Repo description
        const description = document.createElement('p');
        description.textContent = repo.description || 'No description provided.';
        // Stats list
        const statsList = document.createElement('ul');
        statsList.className = 'tech-list';
        // Primary language
        if (repo.language) {
          const langItem = document.createElement('li');
          langItem.textContent = repo.language;
          statsList.appendChild(langItem);
        }
        // Stars
        const starsItem = document.createElement('li');
        starsItem.innerHTML = `<i class="fa fa-star"></i> ${repo.stargazers_count}`;
        statsList.appendChild(starsItem);
        // Forks
        const forksItem = document.createElement('li');
        forksItem.innerHTML = `<i class="fa fa-code-branch"></i> ${repo.forks_count}`;
        statsList.appendChild(forksItem);
        // Link to repository
        const link = document.createElement('a');
        link.href = repo.html_url;
        link.target = '_blank';
        link.rel = 'noopener';
        link.className = 'contact-item';
        link.style.marginTop = '1rem';
        link.innerHTML = '<i class="fa fa-external-link-alt"></i> View on GitHub';

        card.appendChild(title);
        card.appendChild(description);
        card.appendChild(statsList);
        card.appendChild(link);

        projectsContainer.appendChild(card);
      });
    })
    .catch((error) => {
      console.warn('Unable to load GitHub projects:', error);
    });
}

// Load projects after DOM content has loaded
window.addEventListener('DOMContentLoaded', loadGitHubProjects);