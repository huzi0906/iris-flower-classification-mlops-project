# GitHub Issues for Sprint Planning Guide

This document provides guidelines for using GitHub Issues to manage sprint planning and task tracking for the MLOps project.

## Setup

### 1. Create Issue Labels

Create the following labels in your GitHub repository:

- `epic`: High-level features that span multiple sprints
- `user-story`: User stories to be completed within a sprint
- `task`: Individual tasks within a user story
- `bug`: Issues that represent bugs or defects
- `documentation`: Documentation-related tasks
- `enhancement`: Feature enhancements
- `infrastructure`: Infrastructure-related tasks
- `technical-debt`: Technical debt tasks
- `priority:high`: High priority items
- `priority:medium`: Medium priority items
- `priority:low`: Low priority items
- `sprint:1`, `sprint:2`, etc.: Labels for assigning issues to sprints
- `status:blocked`: Issues that are blocked
- `status:in-progress`: Issues currently being worked on
- `status:review`: Issues ready for review

### 2. Create Milestones

Create milestones for each sprint with specific start and end dates. For example:

- **Sprint 1**: Setting up the project infrastructure (May 13-19, 2025)
- **Sprint 2**: Data processing pipeline and model training (May 20-26, 2025)
- **Sprint 3**: API development and testing (May 27-June 2, 2025)
- **Sprint 4**: CI/CD pipeline and deployment (June 3-9, 2025)

## Sprint Planning Process

### Step 1: Create Epics

Create epic issues that represent major features or work streams. For example:

```
Title: [EPIC] End-to-End MLOps Pipeline for Iris Classification

Description:
Implement a complete MLOps pipeline for Iris Classification including data processing, model training,
experiment tracking, and deployment to production.

Epic Tasks:
- Set up project structure and infrastructure
- Implement data processing pipeline
- Build model training workflow
- Create model serving API
- Set up CI/CD pipeline
- Deploy to Kubernetes
```

### Step 2: Break Down Epics into User Stories

Create user stories that can be completed within a sprint. For example:

```
Title: [USER STORY] As a data scientist, I want to track my model experiments to compare different parameters

Description:
As a data scientist, I want to track my model training experiments to easily compare different
hyperparameters and their impact on model performance metrics.

Acceptance Criteria:
- MLflow tracking is configured
- Model parameters are logged
- Model metrics are logged
- Model artifacts are saved
- Experiments can be compared via the MLflow UI

Related Epic: #1 (link to the epic issue)
```

### Step 3: Break Down User Stories into Tasks

Create task issues for individual development tasks. For example:

```
Title: Set up MLflow tracking server

Description:
Configure MLflow tracking server to store experiment metrics and artifacts.

Tasks:
- [ ] Install MLflow
- [ ] Configure MLflow backend store
- [ ] Configure MLflow artifact store
- [ ] Create script to start MLflow server
- [ ] Document setup process

Related User Story: #5 (link to the user story)
```

### Step 4: Run Sprint Planning Meeting

1. **Preparation**:

   - Review and prioritize the backlog
   - Define sprint goals

2. **Meeting Agenda**:

   - Review previous sprint (if applicable)
   - Discuss sprint goals
   - Select user stories for the sprint
   - Break down user stories into tasks if needed
   - Assign tasks to team members
   - Estimate complexity/effort for each task

3. **After the Meeting**:
   - Update GitHub Issues with assignments and sprint labels
   - Move selected issues to the sprint milestone
   - Create a sprint planning document summarizing decisions

## Daily Stand-ups

Use GitHub Issues to track progress during daily stand-ups:

1. Update issue statuses using labels: `status:in-progress`, `status:review`, etc.
2. Comment on issues with updates
3. Document blockers and dependencies

## Sprint Review and Retrospective

After each sprint:

1. **Sprint Review**:

   - Close completed issues
   - Document completed work
   - Demo new features/functionality

2. **Retrospective**:
   - Create a retrospective document
   - Note what went well
   - Note what could be improved
   - Define action items for process improvements

## Example Sprint Board Using GitHub Projects

Create a GitHub Project board with the following columns:

- **Backlog**: All unscheduled issues
- **Sprint Backlog**: Issues scheduled for the current sprint
- **In Progress**: Issues currently being worked on
- **Review**: Issues ready for review
- **Done**: Completed issues

Configure automation rules:

- When issues are labeled with `status:in-progress`, move to "In Progress"
- When issues are labeled with `status:review`, move to "Review"
- When issues are closed, move to "Done"

## Sprint Documentation Template

For each sprint, create an issue with the following template:

```
# Sprint X Planning (DATES)

## Sprint Goals
- [Goal 1]
- [Goal 2]
- [Goal 3]

## User Stories
- [ ] #XX [User Story Title]
- [ ] #XX [User Story Title]
- [ ] #XX [User Story Title]

## Team Capacity
- Team Member 1: X story points
- Team Member 2: X story points
- Team Member 3: X story points

## Risks and Dependencies
- [Risk 1]
- [Dependency 1]

## Definition of Done
- Code complete
- Tests written and passing
- Documentation updated
- PR approved and merged
- Deployed to test environment
```

## Sample Sprint 1 Planning

Here's a sample sprint planning for Sprint 1:

```
# Sprint 1 Planning (May 13-19, 2025)

## Sprint Goals
- Set up project structure and infrastructure
- Implement initial data processing pipeline
- Configure version control for data and code

## User Stories
- [ ] #1 As a developer, I want a structured project setup to easily navigate and contribute to the codebase
- [ ] #2 As a data scientist, I want to process the Iris dataset to prepare it for model training
- [ ] #3 As a developer, I want to track data versions to maintain reproducibility

## Team Capacity
- Alice: 8 story points
- Bob: 8 story points
- Charlie: 8 story points

## Risks and Dependencies
- Team is new to DVC for data versioning
- Need to define consistent coding standards

## Definition of Done
- Code complete
- Tests written and passing
- Documentation updated
- PR approved and merged
- Deployed to test environment
```
