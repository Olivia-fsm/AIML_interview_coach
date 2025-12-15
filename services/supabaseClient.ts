import { createClient } from '@supabase/supabase-js';

// 1. Project URL (API URL)
// Derived from your dashboard link: https://supabase.com/dashboard/project/ufznhjmoheldmtvpomes
const supabaseUrl = 'https://ozuaqgxgnpfiqmctatdr.supabase.co';

// 2. Anon Key (API Key)
// You MUST copy this from: Supabase Dashboard -> Project Settings -> API -> Project API keys -> anon / public
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im96dWFxZ3hnbnBmaXFtY3RhdGRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU4MDYwNTAsImV4cCI6MjA4MTM4MjA1MH0.qyBfUyoNWbqjM4tUZPYMQYQ4u-UuWh85vMG8WFKsLZc';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);