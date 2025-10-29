#!/bin/bash

# Ethereum Graph Analysis - Vercel Deployment Script
echo "🚀 Deploying Ethereum Graph Analysis Dashboard to Vercel..."

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if user is logged in
if ! vercel whoami &> /dev/null; then
    echo "🔐 Please login to Vercel first:"
    echo "Run: vercel login"
    echo "Then run this script again."
    exit 1
fi

# Deploy to production
echo "📦 Deploying to production..."
vercel --prod --yes

echo "✅ Deployment complete!"
echo "🌐 Your dashboard is now live!"
echo "📊 Check the URL provided above to view your Ethereum analysis dashboard"
