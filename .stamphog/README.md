# .stamphog

Configuration for stamphog, the AI reviewer that can approve pull requests in this repository.

Stamphog is a hosted PostHog product. This repository does not run it.
A GitHub App sends the pull request webhooks of this repository to PostHog, the review runs in a sandbox there, and stamphog posts the verdict back to the PR.
The engine and the per-repository settings (enabled, review mode, trigger label) are in [`products/stamphog/`](https://github.com/PostHog/posthog/tree/master/products/stamphog) in the PostHog monorepo.

## Request a review

Add the `stamphog` label to a PR that is not a draft.
An approval is a real GitHub review by `stamphog[bot]`.
Every other verdict is a comment review.
A refusal or an escalation removes the label, so you can add it again after you address the feedback.
A `WAIT` or `ERROR` verdict keeps the label, and the next push tries again.

## Configuration

Stamphog reads this directory from `master`, not from the PR head, so a PR cannot change the rules that judge it.

This directory has no `policy.yml` and no `review-guidance.md`, so reviews use the hosted default policy and the hosted review norms.
The hosted deny-list fits this repository: on 2026-10-01, the only tracked file that matched a deny category was `requirements.txt` (`deps_toolchain`).

[`steering.md`](steering.md) names three risks that are particular to this repository and that the hosted norms do not name: model outputs, training and evaluation data, and the Jev-compatible API.
Stamphog appends it to the hosted review norms.

Steering is the right place for these risks, and `policy.yml` is not.
Each section that `policy.yml` declares replaces the hosted section wholesale.
To deny `data/manifest.json`, this repository would have to copy the full `deny` section, and that copy would stop getting upstream fixes.

What each file can contain: [the engine's "Policy files" section](https://github.com/PostHog/posthog/blob/master/products/stamphog/packages/pr-approval-agent/README.md#policy-files).

## Change this directory

Every path in `.stamphog/` matches the `stamphog_policy` deny category, so stamphog never approves a change to its own configuration.
A human must review it.
