import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import certifi

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

async def seed():
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    
    print("Clearing existing emails...")
    await db.emails.delete_many({})
    await db.participants.delete_many({})

    emails = [
        # ============================================================
        # EMAIL 0: WELCOME (Always first, non-experimental)
        # ============================================================
        {
            "sender_name": "Prolific Research Team",
            "sender_email": "support@prolific.co",
            "subject": "Welcome to the Study: Email Decision Making",
            "body": """
            <p>Dear Participant,</p>
            <p>Thank you for joining our research on workplace email decision-making.</p>
            <br/>
            <p><strong>What to expect:</strong></p>
            <ul>
                <li>You'll review 16 workplace emails.</li>
                <li>Decide how you would respond to each one.</li>
                <li>Your decisions help us understand email behavior patterns.</li>
            </ul>
            <p>All emails are simulated for research purposes. Take your time and respond naturally, as you would in your actual workplace.</p>
            <p>Good luck.</p>
            <br/>
            <p>Best regards,<br>Principal Investigator</p>
            """,
            "is_phishing": False,
            "order_id": 0,
            "experimental": False,
        },

        # ============================================================
        # SHUFFLED EXPERIMENTAL EMAILS
        # P = Phishing, L = Legitimate
        # ============================================================

        # ORDER 1: [P] CEO Fraud - High aggression opener
        {
            "sender_name": "James Anderson (CEO)",
            "sender_email": "j.anderson@netflix-executive.com",
            "subject": "URGENT: Critical Vendor Payment - Board Meeting in Progress",
            "body": """
            <p>Hi,</p>
            <p>I'm in an emergency board meeting and can't access my regular email.</p>
            <p><strong>We need to process a vendor payment of $48,000 TODAY</strong> or we risk losing our biggest contract renewal. The vendor's legal team is threatening to terminate unless payment arrives by 5 PM.</p>
            <p>I've already approved this with Finance. Please process the wire transfer immediately using this secure portal: <a href="http://secure-payment-gateway.net/urgent-transfer" style="color: #4287f5; text-decoration: underline; font-weight: bold;">Process Emergency Payment</a></p>
            <p>This is time-sensitive. I'll be unavailable for the next 3 hours, so I need you to handle this.</p>
            <p>The board will ask about this tomorrow - we cannot afford delays.</p>
            <br/>
            <p>Thanks,<br>James<br>Sent from <u style="color: #4287f5;">Outlook for iOS</u></p>
            """,
            "is_phishing": True,
            "order_id": 1,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "high", "framing": "threat"},
            "aggression_level": "very_high",
            "tactics": ["authority_pressure", "time_urgency", "financial_threat", "isolation_from_verification"],
        },

        # ORDER 2: [L] CFO Budget Deadline
        {
            "sender_name": "Jennifer Wu (CFO)",
            "sender_email": "j.wu@amazon.com",
            "subject": "URGENT: Q4 Budget Submission Due TODAY at 5 PM",
            "body": """
            <p>Dear All Department Heads,</p>
            <p>This is your final reminder that <strong>Q4 budget submissions are due TODAY by 5:00 PM</strong> without exception.</p>
            <p>Late submissions will result in your department being allocated last year's budget with no adjustments, which could significantly impact your operational capacity for Q4.</p>
            <p>Submit your budget here: <a href="https://budget.amazon.com/q4-submissions" style="color: #0078d4; text-decoration: underline;">Q4 Budget Portal</a></p>
            <p>If you have questions, contact Finance immediately - I have meetings from 2-5 PM but my team can assist.</p>
            <p>This deadline is firm due to Board reporting requirements.</p>
            <br/>
            <p>Best regards,<br>Jennifer Wu<br>Chief Financial Officer<br>Direct: (555) 0192</p>
            """,
            "is_phishing": False,
            "order_id": 2,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "high", "framing": "threat"},
            "aggression_level": "high",
            "tactics": ["real_deadline", "legitimate_consequences", "organizational_pressure"],
        },

        # ORDER 3: [P] HR Bonus
        {
            "sender_name": "Sarah Martinez (HR)",
            "sender_email": "smartinez@rnicrosoft.com",
            "subject": "Surprise Employee Appreciation Award - Claim by 6 PM Today!",
            "body": """
            <p>Great news!</p>
            <p>You've been selected by leadership for an <strong>Employee Excellence Spot Bonus</strong> of $750 as part of our Q4 recognition program.</p>
            <p>To claim your award, please verify your direct deposit information before the 6 PM cutoff today (Finance deadline):</p>
            <p><a href="http://hr-rewards-microsoft.com/claim/employee-bonus" style="color: #2e7d32; text-decoration: underline; font-weight: bold;">Claim Your Bonus Now</a></p>
            <p><em>Note: Unclaimed bonuses after today will be reallocated to the general fund. Don't miss out!</em></p>
            <br/>
            <p>Congratulations on your hard work!</p>
            <br/>
            <p>Best regards,<br>Sarah Martinez<br>Human Resources</p>
            """,
            "is_phishing": True,
            "order_id": 3,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "high", "framing": "reward"},
            "aggression_level": "high",
            "tactics": ["reward_greed", "false_urgency", "scarcity", "impersonation"],
        },

        # ORDER 4: [L] Benefits Enrollment
        {
            "sender_name": "Rachel Kim (HR Benefits)",
            "sender_email": "r.kim@ibm.com",
            "subject": "Open Enrollment Ends Friday - New HSA Employer Match!",
            "body": """
            <p>Hi Everyone,</p>
            <p><strong>Important: Open enrollment closes this Friday at 11:59 PM.</strong></p>
            <p>NEW THIS YEAR: We're introducing a <strong>50% employer HSA match up to $1,000</strong> - free money for your healthcare expenses!</p>
            <p>Don't miss out on:</p>
            <ul>
                <li>Enhanced health plan options</li>
                <li>HSA employer contribution ($1,000 match)</li>
                <li>Improved dental/vision coverage</li>
                <li>New wellness program with $200 rewards</li>
            </ul>
            <p>Review and enroll: <a href="https://benefits.ibm.com/enrollment" style="color: #2e7d32; text-decoration: underline;">Benefits Enrollment Portal</a></p>
            <p>Questions? I'm hosting a live Q&A Thursday at 12 PM: <a href="https://meet.ibm.com/benefits-qa" style="color: #0078d4; text-decoration: underline;">Join Meeting</a></p>
            <p>Rachel Kim<br>Benefits Coordinator<br>HR@ibm.com</p>
            """,
            "is_phishing": False,
            "order_id": 4,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "high", "framing": "reward"},
            "aggression_level": "medium",
            "tactics": ["real_benefit", "time_sensitive_enrollment", "legitimate_opportunity"],
        },

        # ORDER 5: [P] Microsoft Account Locked
        {
            "sender_name": "Microsoft 365 Security",
            "sender_email": "security-alerts@microsoft-account-protection.com",
            "subject": "URGENT: Unusual sign-in activity detected - Account locked",
            "body": """
            <div style="font-family: Segoe UI, sans-serif; max-width: 600px;">
                <div style="background: #0078d4; color: white; padding: 20px;">
                    <h2 style="margin: 0;">Microsoft 365</h2>
                </div>
                <div style="padding: 20px; background: #fff3cd; border: 2px solid #ff9800;">
                    <h3 style="color: #d32f2f; margin-top: 0;">Security Alert: Unauthorized Access Detected</h3>
                    <p><strong>We detected a sign-in attempt from an unrecognized device:</strong></p>
                    <ul>
                        <li>Location: Lagos, Nigeria</li>
                        <li>Time: 3:42 AM EST</li>
                    </ul>
                    <p style="color: #d32f2f;"><strong>Your account has been temporarily locked to prevent unauthorized access.</strong></p>
                </div>
                <div style="padding: 10px;">
                    <p>You must verify your identity within <strong>24 hours</strong> or your account will be permanently disabled.</p>
                    <p><a href="http://microsoft-security-verification.com/recover" style="display: inline-block; background: #0078d4; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Verify Your Identity Now</a></p>
                    <p style="font-size: 12px; color: #666;">If you don't recognize this activity, your password may be compromised.</p>
                </div>
                <div style="background: #f5f5f5; padding: 15px; font-size: 11px; color: #666;">
                    Microsoft Corporation | One Microsoft Way, Redmond, WA 98052
                </div>
            </div>
            """,
            "is_phishing": True,
            "order_id": 5,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "high", "framing": "threat"},
            "aggression_level": "very_high",
            "tactics": ["fear_urgency", "brand_impersonation", "sophisticated_design", "account_lockout_threat"],
        },

        # ORDER 6: [L] Team Lunch (Breather after intense emails)
        {
            "sender_name": "David Park (Team Lead)",
            "sender_email": "d.park@google.com",
            "subject": "Team Lunch This Friday - Pizza Party?",
            "body": """
            <p>Hello team,</p>
            <p>Wanted to see if everyone's available for lunch this Friday around 12:30?</p>
            <p>Let's celebrate hitting our Q3 targets early! I'm thinking we order pizza from that place everyone loves.</p>
            <p>Here's the menu so you can think about what you want: <a href="https://www.dominos.com/menu" style="color: #0078d4; text-decoration: underline;">Domino's Menu</a></p>
            <p>My treat! Let me know by Wednesday so I can place the order.</p>
            <p>Also, if anyone has dietary restrictions I should know about, give me a heads up.</p>
            <br/>
            <p>Cheers,<br>David</p>
            """,
            "is_phishing": False,
            "order_id": 6,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["team_building", "casual_communication", "social_reward"],
        },

        # ORDER 7: [P] Security Training
        {
            "sender_name": "IT Security Team",
            "sender_email": "security@cloudflare.com",
            "subject": "Annual Security Awareness Training - Completion Required",
            "body": """
            <p>Hello,</p>
            <p>Our records show you have not yet completed your mandatory Annual Security Awareness Training.</p>
            <p>Please complete this training module by the end of the month to maintain your system access privileges:</p>
            <p><a href="http://cloudflare-training-portal.net/security-module" style="color: #0078d4; text-decoration: underline;">Access Training Module</a></p>
            <p>Estimated completion time: 20 minutes</p>
            <p>Thank you for helping us maintain a secure workplace.</p>
            <p>IT Security Team<br>Cloudflare Inc.</p>
            """,
            "is_phishing": True,
            "order_id": 7,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["routine_mimicry", "compliance_exploitation", "legitimate_appearance"],
        },

        # ORDER 8: [L] Password Update
        {
            "sender_name": "IT Department",
            "sender_email": "it-support@t-mobile.com",
            "subject": "Reminder: Quarterly Password Update Required by Month End",
            "body": """
            <p>Hello,</p>
            <p>As part of our security policy, all employees must update their network passwords every 90 days.</p>
            <p>Your current password expires on the 30th of this month. Please update it before then to avoid any access disruptions.</p>
            <p>To change your password:</p>
            <ol>
                <li>Go to <a href="https://portal.t-mobile.com/password-change" style="color: #0078d4; text-decoration: underline;">Employee Portal</a></li>
                <li>Click "Change Password" under Account Settings</li>
                <li>Follow the prompts (password must be 12+ characters with mixed case, numbers, and symbols)</li>
            </ol>
            <p>You'll receive another reminder next week if your password hasn't been updated.</p>
            <p>If you experience any issues, contact the IT Help Desk at ext. 4357.</p>
            <p>Thank you,<br>IT Support Team<br>T-Mobile</p>
            """,
            "is_phishing": False,
            "order_id": 8,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["routine_maintenance", "policy_compliance", "help_desk_contact"],
        },

        # ORDER 9: [P] Amazon Prize
        {
            "sender_name": "Amazon Customer Service",
            "sender_email": "customer-rewards@amazon-loyalty-program.com",
            "subject": "Congratulations! You've Won a $500 Amazon Gift Card",
            "body": """
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background: #ff9900; padding: 20px; text-align: center;">
                    <h1 style="color: white; margin: 0;">amazon</h1>
                </div>
                <div style="padding: 30px; background: white;">
                    <h2 style="color: #ff9900;">You're a Winner!</h2>
                    <p>Dear Valued Customer,</p>
                    <p>You've been randomly selected from our <strong>Prime Member Loyalty Program</strong> to receive a <span style="color: #ff9900; font-weight: bold;">$500 Amazon Gift Card</span>!</p>
                    <p><strong>To claim your reward, you must act within 24 hours:</strong></p>
                    <ol>
                        <li>Verify your account details</li>
                        <li>Confirm your mailing address</li>
                        <li>Choose digital or physical gift card</li>
                    </ol>
                    <p style="text-align: center; margin: 30px 0;">
                        <a href="http://amazon-rewards-claim.com/verify/gift500" style="background: #ff9900; color: white; padding: 15px 40px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">CLAIM YOUR $500 REWARD</a>
                    </p>
                    <p style="color: #d32f2f; font-size: 14px;"><em>This offer expires in 23 hours. Unclaimed rewards will be forfeited.</em></p>
                </div>
            </div>
            """,
            "is_phishing": True,
            "order_id": 9,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "high", "framing": "reward"},
            "aggression_level": "high",
            "tactics": ["greed_exploitation", "false_scarcity", "brand_trust_abuse", "too_good_to_be_true"],
        },

        # ORDER 10: [L] Vendor License Renewal
        {
            "sender_name": "CrowdStrike Support",
            "sender_email": "support@crowdstrike.com",
            "subject": "ACTION REQUIRED: Software License Renewal Due in 48 Hours",
            "body": """
            <p>Dear Customer,</p>
            <p>Our records indicate that your enterprise software licenses for <strong>Falcon Endpoint Protection</strong> (50 user licenses) expire in 48 hours on January 15, 2025.</p>
            <p><strong style="color: #d32f2f;">Failure to renew will result in immediate service interruption and endpoint protection loss.</strong></p>
            <p>To avoid disruption:</p>
            <ol>
                <li>Review your renewal quote: <a href="https://portal.crowdstrike.com/renewals/INV-2025-4829" style="color: #0078d4; text-decoration: underline;">View Quote #INV-2025-4829</a></li>
                <li>Contact your account manager: sarah.johnson@crowdstrike.com</li>
                <li>Process payment via portal or wire transfer</li>
            </ol>
            <p><strong>Your account details:</strong></p>
            <ul>
                <li>Account ID: AC-94821</li>
                <li>License Count: 50 users</li>
                <li>Renewal Amount: $8,750/year</li>
            </ul>
            <p>If you've already renewed, please disregard this notice.</p>
            <p>CrowdStrike Customer Success<br>support@crowdstrike.com | 1-800-555-TECH</p>
            """,
            "is_phishing": False,
            "order_id": 10,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "high", "framing": "threat"},
            "aggression_level": "high",
            "tactics": ["legitimate_vendor_urgency", "service_disruption_warning", "real_business_consequence"],
        },

        # ORDER 11: [P] DocuSign
        {
            "sender_name": "DocuSign Notification",
            "sender_email": "notifications@docusign-docs.net",
            "subject": "Document Awaiting Your Signature - Reminder",
            "body": """
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background: #f5f5f5; padding: 15px; border-bottom: 3px solid #ffcd00;">
                    <h2 style="color: #5b5b5b; margin: 0;">DocuSign</h2>
                </div>
                <div style="padding: 25px;">
                    <p>Hello,</p>
                    <p>You have a document waiting for your review and signature:</p>
                    <div style="background: #f9f9f9; padding: 15px; margin: 20px 0; border-left: 4px solid #ffcd00;">
                        <strong>Document:</strong> Updated_Employment_Agreement.pdf<br>
                        <strong>From:</strong> HR Department (hr@yourcompany.com)<br>
                        <strong>Expires:</strong> 7 days
                    </div>
                    <p>Please review and sign the document at your earliest convenience:</p>
                    <p><a href="http://docusign-review-portal.com/document/827491" style="background: #5b5b5b; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; display: inline-block;">Review Document</a></p>
                    <p style="font-size: 12px; color: #888; margin-top: 30px;">This is an automated reminder. You will receive additional reminders until the document is completed or expires.</p>
                </div>
            </div>
            """,
            "is_phishing": True,
            "order_id": 11,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "low", "framing": "threat"},
            "aggression_level": "medium",
            "tactics": ["service_impersonation", "professional_appearance", "routine_request"],
        },

        # ORDER 12: [L] Conference Discount
        {
            "sender_name": "Global Cyber Conference Bureau",
            "sender_email": "registrations@globalcyberconference2025.org",
            "subject": "Last Day for Early Bird Discount - Global Cyber Conference 2025",
            "body": """
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background: #1a237e; color: white; padding: 20px;">
                    <h2 style="margin: 0;">Global Cyber Conference 2025</h2>
                    <p style="margin: 5px 0 0 0;">March 15-17 | San Francisco</p>
                </div>
                <div style="padding: 25px;">
                    <h3 style="color: #d32f2f;">Final Day for Early Bird Pricing!</h3>
                    <p>Don't miss your chance to save <strong>$400 on registration</strong> for the industry's premier cybersecurity conference.</p>
                    <div style="background: #e8f5e9; padding: 15px; margin: 20px 0; border-left: 4px solid #4caf50;">
                        <strong>Early Bird: $599</strong> (ends tonight at 11:59 PM)<br>
                        Regular Price: $999 (starts tomorrow)
                    </div>
                    <p><strong>Featured speakers include:</strong></p>
                    <ul>
                        <li>Dr. Emily Chen, CISO at Fortune 500</li>
                        <li>James Rodriguez, FBI Cyber Division</li>
                        <li>Sarah Thompson, PhD, MIT Security Lab</li>
                    </ul>
                    <p style="text-align: center; margin: 25px 0;">
                        <a href="https://register.globalcyberconference2025.org/early-bird" style="background: #4caf50; color: white; padding: 15px 35px; text-decoration: none; border-radius: 4px; font-weight: bold; display: inline-block;">REGISTER NOW - SAVE $400</a>
                    </p>
                    <p style="font-size: 13px; color: #666;">Group discounts available for 5+ attendees from the same organization.</p>
                </div>
            </div>
            """,
            "is_phishing": False,
            "order_id": 12,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "high", "framing": "reward"},
            "aggression_level": "medium",
            "tactics": ["legitimate_time_discount", "professional_opportunity", "real_event"],
        },

        # ORDER 13: [P] LinkedIn Learning
        {
            "sender_name": "LinkedIn Learning",
            "sender_email": "learning@linkedln-courses.com",
            "subject": "Your Free Premium Course Access Has Been Activated",
            "body": """
            <p><strong>LinkedIn Learning</strong></p>
            <p>Good news - you now have complimentary access to our Premium course library!</p>
            <p>This benefit includes:</p>
            <ul>
                <li>16,000+ expert-led courses</li>
                <li>Personalized recommendations</li>
                <li>Certificates of completion</li>
                <li>Offline viewing</li>
            </ul>
            <p>Start learning today: <a href="http://linkedin-learning-premium.com/activate" style="color: #0077b5; text-decoration: underline;">Activate Your Premium Access</a></p>
            <p>This promotional offer is available for a limited time based on your professional profile.</p>
            <p>Happy learning!</p>
            """,
            "is_phishing": True,
            "order_id": 13,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["professional_development_lure", "free_benefit", "subtle_domain_typo"],
        },

        # ORDER 14: [L] Backup Report
        {
            "sender_name": "McAfee Automated Monitoring",
            "sender_email": "notifications@mcafee.com",
            "subject": "Monthly Data Backup Report - Action Items",
            "body": """
            <p>Hello Administrator,</p>
            <p>Your monthly backup report for December 2024 is ready. Overall system health: <strong style="color: #4caf50;">Good</strong></p>
            <p><strong>Items requiring attention:</strong></p>
            <ul>
                <li>Database backup completed with 2 warnings (non-critical)</li>
                <li>3 files skipped due to open file locks</li>
                <li>Storage capacity at 78% - consider expansion within 60 days</li>
            </ul>
            <p>View full report: <a href="https://portal.mcafee.com/reports/DEC2024" style="color: #0078d4; text-decoration: underline;">December 2024 Report</a></p>
            <p><strong>Recommended actions:</strong></p>
            <ol>
                <li>Review skipped files and schedule backup during maintenance window</li>
                <li>Plan storage expansion before reaching 85% capacity</li>
            </ol>
            <p>No immediate action required. Your next backup is scheduled for January 15, 2025.</p>
            <p>Questions? Contact support@mcafee.com</p>
            <p>McAfee Data Protection<br>Automated Monitoring System</p>
            """,
            "is_phishing": False,
            "order_id": 14,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["routine_monitoring", "maintenance_notification", "service_provider_communication"],
        },

        # ORDER 15: [L] Newsletter
        {
            "sender_name": "Cybersecurity Today",
            "sender_email": "newsletter@cybersecuritytoday.com",
            "subject": "This Week in Cybersecurity: New Ransomware Trends & Best Practices",
            "body": """
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background: #263238; color: white; padding: 20px;">
                    <h2 style="margin: 0;">Cybersecurity Today</h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Weekly Industry Newsletter</p>
                </div>
                <div style="padding: 25px;">
                    <h3>Top Stories This Week:</h3>
                    <div style="border-left: 3px solid #0078d4; padding-left: 15px; margin: 20px 0;">
                        <p><strong>Ransomware Trends:</strong> New variants targeting healthcare sector with double-extortion tactics. <a href="https://cybersecuritytoday.com/articles/ransomware-trends-2025" style="color: #0078d4;">Read more →</a></p>
                    </div>
                    <div style="border-left: 3px solid #4caf50; padding-left: 15px; margin: 20px 0;">
                        <p><strong>Best Practice Guide:</strong> Implementing zero-trust architecture - A step-by-step checklist. <a href="https://cybersecuritytoday.com/guides/zero-trust" style="color: #0078d4;">Read more →</a></p>
                    </div>
                    <div style="border-left: 3px solid #ff9800; padding-left: 15px; margin: 20px 0;">
                        <p><strong>Upcoming Webinar:</strong> "Phishing Defense Strategies for 2025" - Jan 25, 2 PM EST (Free). <a href="https://cybersecuritytoday.com/webinars/phishing-defense" style="color: #0078d4;">Register →</a></p>
                    </div>
                    <p style="margin-top: 30px; font-size: 13px; color: #666;">
                        You're receiving this because you subscribed to our newsletter.<br>
                        <a href="https://cybersecuritytoday.com/preferences" style="color: #666;">Manage preferences</a> | 
                        <a href="https://cybersecuritytoday.com/unsubscribe" style="color: #666;">Unsubscribe</a>
                    </p>
                </div>
            </div>
            """,
            "is_phishing": False,
            "order_id": 15,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["informational_content", "professional_newsletter", "opt_in_communication"],
        },

        # ============================================================
        # ORDER 16: [P] PROLIFIC PHISHING - MUST BE LAST!
        # Critical: Tests post-study trust exploitation
        # ============================================================
        {
            "sender_name": "Prolific Research Team",
            "sender_email": "opportunities@prolific-studies.com",
            "subject": "Thank You + More High-Paying Studies Available!",
            "body": """
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background: #6d3ef2; color: white; padding: 20px;">
                    <h2 style="margin: 0;">Prolific</h2>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">Participant Opportunities</p>
                </div>
                <div style="padding: 25px;">
                    <p>Hi there,</p>
                    <p>Thank you for completing the <strong>"Email Decision Making"</strong> study! Your responses were excellent and your payment has been processed.</p>
                    <p style="background: #f0f7ff; padding: 15px; border-left: 4px solid #6d3ef2; margin: 20px 0;">
                        <strong>Good news:</strong> Based on your profile and performance, you've been selected for our <strong>Premium Participant Program</strong> with access to higher-paying studies.
                    </p>
                    <p>We have <strong>3 exclusive opportunities</strong> available right now that match your demographics:</p>
                    <ul style="line-height: 1.8;">
                        <li><strong>Consumer Behavior Study</strong> - £18 for 40 minutes</li>
                        <li><strong>Technology Usage Survey</strong> - £15 for 30 minutes</li>
                        <li><strong>Workplace Decision Making (Part 2)</strong> - £22 for 45 minutes</li>
                    </ul>
                    <p><strong>These spots fill quickly!</strong> Click here to view and claim your preferred studies:</p>
                    <p style="text-align: center; margin: 25px 0;">
                        <a href="http://prolific-premium-studies.com/claim" style="background: #6d3ef2; color: white; padding: 15px 35px; text-decoration: none; border-radius: 4px; font-weight: bold; display: inline-block;">→ View My Exclusive Studies</a>
                    </p>
                    <p style="font-size: 13px; color: #d32f2f;"><em>Note: These opportunities are reserved for top-performing participants and expire in 48 hours.</em></p>
                    <p style="margin-top: 30px;">Happy earning!<br><strong>The Prolific Team</strong></p>
                </div>
                <div style="background: #f5f5f5; padding: 15px; font-size: 11px; color: #666;">
                    Prolific Academic Ltd | Helping researchers connect with participants worldwide
                </div>
            </div>
            """,
            "is_phishing": True,
            "order_id": 16,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "low", "framing": "reward"},
            "aggression_level": "medium",
            "tactics": [
                "contextual_trust_exploitation",
                "post_interaction_vulnerability",
                "financial_motivation",
                "exclusivity_scarcity",
                "platform_impersonation",
                "continuation_effect"
            ],
        }
    ]
    
    # Insert all emails
    result = await db.emails.insert_many(emails)
    print(f"Successfully inserted {len(result.inserted_ids)} emails")

if __name__ == "__main__":
    asyncio.run(seed())