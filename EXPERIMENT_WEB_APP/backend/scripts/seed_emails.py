"""
CYPEARL Experiment - Email Seeding Script (UPDATED v3 - FIXED FACTORIAL DESIGN)

FIXES IN THIS VERSION:
1. Corrected factorial design - emails 5-8 now use unknown_external senders
2. All 16 combinations of 2×2×2×2 factorial design are now covered
3. No duplicate combinations
4. All institution names are fictional (no real organizations)
5. Fixed button styling for readability

FACTORIAL DESIGN (2×2×2×2 = 16 combinations):
- Type: Phishing vs Legitimate
- Sender: Known vs Unknown
- Urgency: High vs Low
- Framing: Threat vs Reward

FICTIONAL COMPANY ECOSYSTEM:
- Participant's Employer: Lux Consultancy
- Cloud Provider: NexusCloud
- Professional Network: ProLink
- Security Software: SecureShield
- File Storage: CloudVault
- Productivity Suite: OfficeSuite Pro
- Domain Registrar: DomainHost
- Research Platform: ResearchConnect
- Tax Services: EuroFinance Tax Services (external)
- Business Network: LuxBusiness Network (external)
- Security Newsletter: Cybersecurity Today (external)

Design: 16 experimental emails + 1 welcome email
- 8 phishing, 8 legitimate
- Balanced across urgency (high/low), framing (threat/reward), familiarity (known/unknown)
"""

import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
from motor.motor_asyncio import AsyncIOMotorClient
import certifi


MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "bizmail_db")


def get_seed_emails():
    """
    Return list of email documents for seeding.
    All company names are FICTIONAL to avoid trademark issues.
    Bodies have proper paragraph spacing for realistic email appearance.
    
    FACTORIAL DESIGN MAPPING:
    #1:  Phishing  | Known Internal   | High | Threat
    #2:  Legit     | Known Internal   | High | Threat
    #3:  Phishing  | Known Internal   | High | Reward
    #4:  Legit     | Known Internal   | High | Reward
    #5:  Phishing  | Unknown External | High | Threat  ← FIXED (was known_external)
    #6:  Legit     | Unknown External | High | Threat  ← FIXED (was known_external)
    #7:  Phishing  | Unknown External | High | Reward  ← FIXED (was known_external)
    #8:  Legit     | Unknown External | High | Reward  ← FIXED (was known_external)
    #9:  Phishing  | Known Internal   | Low  | Threat
    #10: Legit     | Known Internal   | Low  | Threat
    #11: Phishing  | Unknown External | Low  | Threat
    #12: Phishing  | Unknown External | Low  | Reward
    #13: Legit     | Known Internal   | Low  | Reward
    #14: Legit     | Unknown External | Low  | Threat
    #15: Legit     | Unknown External | Low  | Reward
    #16: Phishing  | Known Internal   | Low  | Reward
    """
    
    emails = [
        # EMAIL 0: WELCOME EMAIL
        {
            "sender_name": "University of Luxembourg IRiSC Research",
            "sender_email": "irisc@uni.lu",
            "subject": "Welcome to the Study: Email Decision Making",
            
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #222;">

            <!-- Greeting -->
            <div style="margin-bottom: 16px;">
                <p>Dear Participant,</p>
                <p>Thank you for taking part in our research study on workplace email decision-making.</p>
            </div>

            <!-- Your Role -->
            <div style="background: #f8f9ff; padding: 12px 16px; border-left: 4px solid #4f46e5; border-radius: 4px; margin-bottom: 16px;">
                <p style="margin: 0 0 8px 0;"><strong>Your Role</strong></p>
                <p style="margin: 0;">
                    For the purpose of this study, you will briefly <strong>role-play</strong> as an employee at
                    <strong>Lux Consultancy</strong>, a mid-sized consultancy firm. Lux Consultancy works with clients across
                    multiple sectors, including software and SaaS, finance, healthcare, environmental services, human resources,
                    retail, manufacturing, and education.
                </p>
            </div>

            <!-- Company Services -->
            <div style="background: #f4f8fb; padding: 12px 16px; border-left: 4px solid #0ea5e9; border-radius: 4px; margin-bottom: 16px;">
                <p style="margin: 0 0 8px 0;"><strong>Tools and Services You Use</strong></p>
                <p style="margin: 0 0 8px 0;">Your company uses the following internal services:</p>
                <ul style="margin: 0 0 0 20px; padding: 0; list-style-type: disc;">
                    <li style="margin: 2px 0;"><strong>NexusCloud</strong>: Cloud infrastructure</li>
                    <li style="margin: 2px 0;"><strong>ProLink</strong>: Professional networking</li>
                    <li style="margin: 2px 0;"><strong>SecureShield</strong>: Security software</li>
                    <li style="margin: 2px 0;"><strong>CloudVault</strong>: File storage and sharing</li>
                    <li style="margin: 2px 0;"><strong>OfficeSuite Pro</strong>: Productivity and collaboration tools</li>
                </ul>
                <p style="margin: 12px 0 8px 0;">You may also receive emails from external services such as:</p>
                <ul style="margin: 0 0 0 20px; padding: 0; list-style-type: disc;">
                    <li style="margin: 2px 0;"><strong>EuroFinance Tax Services</strong>: Corporate tax preparation</li>
                    <li style="margin: 2px 0;"><strong>LuxBusiness Network</strong>: Business events and networking</li>
                    <li style="margin: 2px 0;"><strong>Cybersecurity Today</strong>: Industry newsletter</li>
                </ul>
            </div>

            <!-- Task / Actions -->
            <div style="background: #f9fafb; padding: 12px 16px; border-left: 4px solid #059669; border-radius: 4px; margin-bottom: 16px;">
                <p style="margin: 0 0 8px 0;"><strong>What You Will Do</strong></p>
                <p style="margin: 0 0 8px 0;">
                    You will review <strong>16 workplace emails</strong> and decide how to respond to each one. For every email, you can:
                </p>
                <ul style="margin: 0 0 0 20px; padding: 0; list-style-type: disc;">
                    <li style="margin: 2px 0;">Mark it as <strong>Safe</strong>: if you believe it is legitimate</li>
                    <li style="margin: 2px 0;"><strong>Report</strong> it: if you suspect it is a phishing attempt</li>
                    <li style="margin: 2px 0;"><strong>Delete</strong> it: if you want to remove it from your inbox</li>
                    <li style="margin: 2px 0;"><strong>Ignore</strong> it: if you are unsure or do not wish to act</li>
                </ul>
            </div>

            <!-- Bonus Points Block (Intro + Rules) -->
            <div style="background: #f0f7ff; padding: 12px 16px; border-left: 4px solid #f97316; border-radius: 4px; margin-bottom: 16px;">

                <!-- Bonus Points Intro -->
                <p style="margin: 0; margin-bottom: 16px;">
                    💰 You can earn <strong style="color: #d97706;">BONUS POINTS</strong> during this study that will be converted into a
                    <strong>monetary bonus</strong> in addition to your base compensation.
                </p>

                <!-- Bonus Points Rules -->
                <p style="margin: 0 0 12px 0;"><strong>👍🏻 Clicking links in LEGITIMATE emails</strong> earns bonus points.</p>
                <p style="margin: 0 0 12px 0;"><strong>👎🏻 Clicking links in PHISHING emails</strong> loses bonus points.</p>
                <p style="margin: 0; background: #fff3cd; padding: 10px; border-radius: 4px; color: #856404;">
                    <strong>⚠️ Important: </strong> You must use your own judgment to decide which links are safe to click. 
                    Your bonus total will be <strong>revealed only at the end</strong> of the study.
                </p>
            </div>

            <!-- Timing / Closing -->
            <div style="margin-bottom: 16px;">
                <p style="margin: 0 0 8px 0;">Please take your time with each message. There is <strong>no time limit</strong> for your decisions.</p>
            </div>

            <div>
                <p style="margin: 0 0 4px 0;">Best regards,</p>
                <p style="margin: 0;">
                    Principal Investigator<br>
                    IRiSC Research Team<br>
                    University of Luxembourg
                </p>
            </div>

        </div>""",

            "is_phishing": False,
            "order_id": 0,
            "experimental": False,
            "email_type": "welcome",
            "sender_familiarity": "known_external",
            "urgency_level": "low",
            "framing_type": "neutral",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "welcome", "sender": "external", "urgency": "low", "framing": "neutral"},
            "aggression_level": "none",
            "tactics": [],
        },

        # EMAIL 1: PHISHING | Known Internal | High Urgency | Threat
        {
            "sender_name": "James Anderson (CEO)",
            "sender_email": "j.anderson@luxconsultancy-executive.com",
            "subject": "URGENT: Critical Vendor Payment - Board Meeting in Progress",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hi,</p>
                
                <p>I'm in an emergency board meeting and can't access my regular email.</p>
                
                <p><strong>We need to process a vendor payment of $48,000 TODAY</strong> or we risk losing our biggest contract renewal.</p>
                
                <p>Please process the wire transfer immediately:</p>
                
                <p><a href="http://secure-payment-gateway.net/urgent-transfer" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Process Emergency Payment</a></p>
                
                <p>This is time-sensitive. I'll be unavailable for 3 hours.</p>
                
                <p>Thanks,</p>
                <p>James</p>
            </div>""",
            "is_phishing": True,
            "order_id": 1,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "known_internal",
            "urgency_level": "high",
            "framing_type": "threat",
            "phishing_quality": "high",
            "has_aggressive_content": True,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "high", "framing": "threat"},
            "aggression_level": "very_high",
            "tactics": ["authority_pressure", "time_urgency", "financial_threat"],
        },

        # EMAIL 2: LEGITIMATE | Known Internal | High Urgency | Threat
        {
            "sender_name": "Jennifer Wu (CFO)",
            "sender_email": "j.wu@luxconsultancy.com",
            "subject": "URGENT: Q4 Budget Submission Due TODAY at 5 PM",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear All Department Heads,</p>
                
                <p>This is your final reminder that <strong>Q4 budget submissions are due TODAY by 5:00 PM</strong>.</p>
                
                <p>Late submissions will result in your department being allocated last year's budget with no adjustments.</p>
                
                <p>Submit your budget here:</p>
                
                <p><a href="https://budget.luxconsultancy.com/q4-submissions" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Q4 Budget Portal</a></p>
                
                <p>Best regards,</p>
                <p>Jennifer Wu<br>Chief Financial Officer<br>Lux Consultancy</p>
            </div>""",
            "is_phishing": False,
            "order_id": 2,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "known_internal",
            "urgency_level": "high",
            "framing_type": "threat",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "high", "framing": "threat"},
            "aggression_level": "high",
            "tactics": ["real_deadline", "legitimate_consequences"],
        },

        # EMAIL 3: PHISHING | Known Internal | High Urgency | Reward
        {
            "sender_name": "Sarah Martinez (HR)",
            "sender_email": "smartinez@luxconsultancy-hr.com",
            "subject": "Surprise Employee Appreciation Award - Claim by 6 PM Today!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Great news!</p>
                
                <p>You've been selected for an <strong>Employee Excellence Spot Bonus</strong> of $750!</p>
                
                <p>To claim your award, verify your direct deposit information before 6 PM:</p>
                
                <p><a href="http://hr-rewards-luxconsultancy.com/claim/employee-bonus" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Claim Your Bonus Now</a></p>
                
                <p><em>Unclaimed bonuses will be reallocated. Don't miss out!</em></p>
                
                <p>Best regards,</p>
                <p>Sarah Martinez<br>Human Resources</p>
            </div>""",
            "is_phishing": True,
            "order_id": 3,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "known_internal",
            "urgency_level": "high",
            "framing_type": "reward",
            "phishing_quality": "high",
            "has_aggressive_content": True,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "high", "framing": "reward"},
            "aggression_level": "high",
            "tactics": ["reward_greed", "false_urgency", "scarcity", "impersonation"],
        },

        # EMAIL 4: LEGITIMATE | Known Internal | High Urgency | Reward
        {
            "sender_name": "David Chen (IT Director)",
            "sender_email": "d.chen@luxconsultancy.com",
            "subject": "Complete Security Training by Friday → Get Extra PTO Day!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hi Team,</p>
                
                <p>Great news! As part of Cybersecurity Awareness Month, anyone who completes the mandatory security training by <strong>this Friday at 5 PM</strong> will receive an extra PTO day.</p>
                
                <p>The training takes about 30 minutes and covers:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Recognizing phishing attempts</li>
                    <li>Password best practices</li>
                    <li>Data handling procedures</li>
                </ul>
                
                <p>Start your training here:</p>
                
                <p><a href="https://training.luxconsultancy.com/security-awareness-2024" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Begin Security Training</a></p>
                
                <p>Best,</p>
                <p>David Chen<br>IT Director<br>Lux Consultancy</p>
            </div>""",
            "is_phishing": False,
            "order_id": 4,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "known_internal",
            "urgency_level": "high",
            "framing_type": "reward",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "high", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["real_incentive", "legitimate_deadline"],
        },

        # EMAIL 5: PHISHING | Unknown External | High Urgency | Threat
        # FIXED: Changed from known_external (NexusCloud) to unknown_external
        {
            "sender_name": "DomainGuard Services",
            "sender_email": "urgent@domaineguard-alerts.com",
            "subject": "⚠️ URGENT: Your Domain luxconsultancy.com Expires in 24 Hours!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="background: #d32f2f; color: white; padding: 12px; margin-bottom: 20px;">
                    <strong>⚠️ DOMAIN EXPIRATION ALERT</strong>
                </div>
                
                <p>Dear Domain Owner,</p>
                
                <p>Our records indicate that your domain <strong>luxconsultancy.com</strong> is scheduled to expire in <strong>24 hours</strong>.</p>
                
                <p>If the domain expires:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Your website will go offline</li>
                    <li>Email services will stop working</li>
                    <li>The domain may be purchased by someone else</li>
                </ul>
                
                <p><strong>Renew immediately to prevent service interruption:</strong></p>
                
                <p><a href="http://domaineguard-alerts.com/renew/luxconsultancy" style="color: #0066cc; background-color: #e6f3ff; padding: 8px 16px; border-radius: 4px; text-decoration: underline; font-weight: bold;">Renew Domain Now - $14.99/year</a></p>
                
                <p style="color: #d32f2f;"><strong>Act now to avoid losing your domain permanently.</strong></p>
                
                <p>DomainGuard Services<br>Customer Support Team</p>
            </div>""",
            "is_phishing": True,
            "order_id": 5,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "unknown_external",
            "urgency_level": "high",
            "framing_type": "threat",
            "phishing_quality": "high",
            "has_aggressive_content": True,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "high", "framing": "threat"},
            "aggression_level": "very_high",
            "tactics": ["fake_service", "domain_scam", "time_pressure", "fear_inducing"],
        },

        # EMAIL 6: LEGITIMATE | Unknown External | High Urgency | Threat
        # FIXED: Changed from known_external (NexusCloud) to unknown_external
        # Using fictional tax/finance service name
        {
            "sender_name": "EuroFinance Tax Services",
            "sender_email": "notifications@eurofinance-tax.eu",
            "subject": "Action Required: Annual Tax Declaration Due January 31st",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear Client,</p>
                
                <p>This is a reminder from EuroFinance Tax Services that your <strong>annual tax declaration for fiscal year 2024</strong> must be submitted by <strong>January 31, 2025</strong>.</p>
                
                <p>As your registered tax preparation service, we want to ensure you meet all deadlines. Failure to submit on time may result in:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Late filing penalties</li>
                    <li>Interest charges on unpaid taxes</li>
                    <li>Estimated assessment by the tax office</li>
                </ul>
                
                <p>Access your tax documents through our client portal:</p>
                
                <p><a href="https://portal.eurofinance-tax.eu/declarations" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Access Tax Portal</a></p>
                
                <p>If you have already submitted your declaration, please disregard this notice.</p>
                
                <p>For assistance, contact our support team at support@eurofinance-tax.eu or call +352 247-5000.</p>
                
                <p>EuroFinance Tax Services<br>Your Trusted Tax Partner</p>
            </div>""",
            "is_phishing": False,
            "order_id": 6,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "unknown_external",
            "urgency_level": "high",
            "framing_type": "threat",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "high", "framing": "threat"},
            "aggression_level": "medium",
            "tactics": ["official_notice", "compliance_deadline"],
        },

        # EMAIL 7: PHISHING | Unknown External | High Urgency | Reward
        # FIXED: Changed from known_external (ProLink) to unknown_external
        # Using fictional business awards name
        {
            "sender_name": "GlobalBiz Excellence Awards",
            "sender_email": "awards@globalbiz-excellence.com",
            "subject": "🏆 Congratulations! Lux Consultancy Nominated for Excellence Award",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%); color: #333; padding: 20px; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0;">🏆 GlobalBiz Excellence Awards 2025</h2>
                </div>
                
                <p>Dear Executive,</p>
                
                <p>We are delighted to inform you that <strong>Lux Consultancy</strong> has been nominated for the <strong>GlobalBiz Excellence Award 2025</strong> in the category of "Outstanding Consultancy Services"!</p>
                
                <p>Benefits of winning include:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Featured in our Business Excellence Special Edition</li>
                    <li>Exclusive networking at the Awards Gala in Brussels</li>
                    <li>Official winner badge for marketing materials</li>
                    <li>€10,000 marketing grant</li>
                </ul>
                
                <p><strong>Accept your nomination within 48 hours to secure your spot:</strong></p>
                
                <p><a href="http://globalbiz-excellence.com/accept-nomination" style="color: #0066cc; background-color: #fff3cd; padding: 8px 16px; border-radius: 4px; text-decoration: underline; font-weight: bold;">Accept Nomination Now</a></p>
                
                <p style="font-size: 13px; color: #666;"><em>A small processing fee of €299 applies to confirm participation.</em></p>
            </div>""",
            "is_phishing": True,
            "order_id": 7,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "unknown_external",
            "urgency_level": "high",
            "framing_type": "reward",
            "phishing_quality": "high",
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "high", "framing": "reward"},
            "aggression_level": "medium",
            "tactics": ["fake_award", "vanity_appeal", "hidden_fees", "time_pressure"],
        },

        # EMAIL 8: LEGITIMATE | Unknown External | High Urgency | Reward
        # FIXED: Changed from known_external (TechConf) to unknown_external
        # Using fictional business network name
        {
            "sender_name": "LuxBusiness Network",
            "sender_email": "events@luxbusiness-network.eu",
            "subject": "Last Chance: Free Seats at Digital Transformation Summit - Tomorrow!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear Business Professional,</p>
                
                <p>Due to late cancellations, we have <strong>5 complimentary seats</strong> available for tomorrow's <strong>Digital Transformation Summit</strong>!</p>
                
                <p><strong>Event Details:</strong></p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li><strong>Date:</strong> Tomorrow, January 15, 2025</li>
                    <li><strong>Time:</strong> 9:00 AM - 5:00 PM</li>
                    <li><strong>Location:</strong> LuxBusiness Conference Center, Kirchberg</li>
                    <li><strong>Value:</strong> €350 (complimentary for registered members)</li>
                </ul>
                
                <p>Sessions include:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>AI Implementation Strategies for SMEs</li>
                    <li>Cybersecurity Best Practices 2025</li>
                    <li>Networking lunch with industry leaders</li>
                </ul>
                
                <p><a href="https://luxbusiness-network.eu/events/digital-summit-2025/register" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Claim Your Free Seat</a></p>
                
                <p>First come, first served. Seats are limited.</p>
                
                <p>Best regards,<br>Events Team<br>LuxBusiness Network</p>
            </div>""",
            "is_phishing": False,
            "order_id": 8,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "unknown_external",
            "urgency_level": "high",
            "framing_type": "reward",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "high", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["legitimate_opportunity", "real_scarcity", "professional_event"],
        },

        # EMAIL 9: PHISHING | Known Internal | Low Urgency | Threat
        {
            "sender_name": "IT Security",
            "sender_email": "itsecurity@security-update-portal.com",
            "subject": "Scheduled Password Expiration Notice",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear Employee,</p>
                
                <p>According to our records, your network password is scheduled to expire in 7 days.</p>
                
                <p>To maintain uninterrupted access to company systems, please update your password at your earliest convenience:</p>
                
                <p><a href="http://security-update-portal.com/password-reset" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Update Password</a></p>
                
                <p>Password requirements:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Minimum 12 characters</li>
                    <li>At least one uppercase letter</li>
                    <li>At least one number</li>
                    <li>At least one special character</li>
                </ul>
                
                <p>If you have any questions, contact the IT Help Desk.</p>
                
                <p>IT Security Team<br>Lux Consultancy</p>
            </div>""",
            "is_phishing": True,
            "order_id": 9,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "known_internal",
            "urgency_level": "low",
            "framing_type": "threat",
            "phishing_quality": "low",
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["routine_impersonation", "credential_harvesting"],
        },

        # EMAIL 10: LEGITIMATE | Known Internal | Low Urgency | Threat
        {
            "sender_name": "Facilities Management",
            "sender_email": "facilities@luxconsultancy.com",
            "subject": "Building Maintenance - Parking Garage Closure Next Week",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear Employees,</p>
                
                <p>Please be advised that the underground parking garage (Levels B1-B3) will be <strong>closed for maintenance</strong> next week:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li><strong>Dates:</strong> Monday, January 13 - Friday, January 17</li>
                    <li><strong>Affected:</strong> All underground levels</li>
                    <li><strong>Reason:</strong> Annual fire suppression system inspection</li>
                </ul>
                
                <p>Alternative parking options:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Surface lot (behind Building C)</li>
                    <li>Street parking (meters free after 6 PM)</li>
                    <li>Public transit subsidy available through HR</li>
                </ul>
                
                <p>We apologize for any inconvenience. Normal operations will resume Monday, January 20.</p>
                
                <p>Facilities Management<br>Lux Consultancy</p>
            </div>""",
            "is_phishing": False,
            "order_id": 10,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "known_internal",
            "urgency_level": "low",
            "framing_type": "threat",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["informational", "advance_notice"],
        },

        # EMAIL 11: PHISHING | Unknown External | Low Urgency | Threat
        {
            "sender_name": "CloudVault Storage",
            "sender_email": "support@cloudvault-secure.net",
            "subject": "Your Storage is Almost Full - Upgrade Available",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Dear User,</p>
                
                <p>Your CloudVault storage is currently at <strong>94% capacity</strong>.</p>
                
                <p>Current plan: Basic (15 GB)<br>
                Used: 14.1 GB<br>
                Remaining: 0.9 GB</p>
                
                <p>When storage is full, you won't be able to:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Upload new files</li>
                    <li>Receive email attachments</li>
                    <li>Sync across devices</li>
                </ul>
                
                <p>Upgrade to Premium for only $2.99/month and get 100 GB:</p>
                
                <p><a href="http://cloudvault-secure.net/upgrade" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Upgrade Now</a></p>
                
                <p>CloudVault Team</p>
            </div>""",
            "is_phishing": True,
            "order_id": 11,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "unknown_external",
            "urgency_level": "low",
            "framing_type": "threat",
            "phishing_quality": "low",
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["service_impersonation", "upsell_scam"],
        },

        # EMAIL 12: PHISHING | Unknown External | Low Urgency | Reward
        {
            "sender_name": "Professional Development",
            "sender_email": "courses@prof-dev-courses.com",
            "subject": "Free Certification Course - Limited Spots Available",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hello Professional,</p>
                
                <p>We're offering a <strong>FREE certification course</strong> in Project Management for qualified candidates.</p>
                
                <p>Course details:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Duration: 6 weeks (self-paced)</li>
                    <li>Value: $1,299 (yours FREE)</li>
                    <li>Certificate: Industry-recognized PMP prep</li>
                    <li>Format: Online with live Q&A sessions</li>
                </ul>
                
                <p>Only 50 spots available. Secure yours now:</p>
                
                <p><a href="http://prof-dev-courses.com/enroll-free" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">Claim Your Free Spot</a></p>
                
                <p>Best regards,</p>
                <p>Professional Development Team</p>
            </div>""",
            "is_phishing": True,
            "order_id": 12,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "unknown_external",
            "urgency_level": "low",
            "framing_type": "reward",
            "phishing_quality": "low",
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "unknown_external", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["too_good_to_be_true", "lead_generation_scam"],
        },

        # EMAIL 13: LEGITIMATE | Known Internal | Low Urgency | Reward
        {
            "sender_name": "HR Team",
            "sender_email": "hr@luxconsultancy.com",
            "subject": "Reminder: Company Anniversary Celebration This Friday!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hi Everyone!</p>
                
                <p>Just a friendly reminder about our <strong>15th Anniversary Celebration</strong> this Friday!</p>
                
                <p><strong>Details:</strong></p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li><strong>Date:</strong> Friday, January 10</li>
                    <li><strong>Time:</strong> 3:00 PM - 6:00 PM</li>
                    <li><strong>Location:</strong> Main Cafeteria & Outdoor Patio</li>
                    <li><strong>Dress code:</strong> Business casual</li>
                </ul>
                
                <p>What to expect:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Catered food and beverages</li>
                    <li>Live music</li>
                    <li>Awards ceremony for long-tenure employees</li>
                    <li>Raffle prizes (including extra PTO days!)</li>
                </ul>
                
                <p>Please RSVP if you haven't already:</p>
                
                <p><a href="https://employee-portal.luxconsultancy.com/events/anniversary-2025" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">RSVP Here</a></p>
                
                <p>Looking forward to celebrating with you!</p>
                
                <p>The HR Team</p>
            </div>""",
            "is_phishing": False,
            "order_id": 13,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "known_internal",
            "urgency_level": "low",
            "framing_type": "reward",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "known_internal", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["genuine_event", "company_culture"],
        },

        # EMAIL 14: LEGITIMATE | Unknown External | Low Urgency | Threat
        {
            "sender_name": "SecureShield Data Protection",
            "sender_email": "reports@secureshield.com",
            "subject": "Monthly Security Report - December 2024",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hello Administrator,</p>
                
                <p>Your monthly security report for December 2024 is ready.</p>
                
                <p>Overall system health: <strong style="color: green;">Good</strong></p>
                
                <p>Items requiring attention:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Database backup completed with 2 warnings (non-critical)</li>
                    <li>3 files skipped due to open file locks</li>
                    <li>Storage capacity at 78% — consider expansion within 60 days</li>
                </ul>
                
                <p>View full report:</p>
                
                <p><a href="https://portal.secureshield.com/reports/DEC2024" style="color: #0066cc; background-color: #e6f3ff; padding: 4px 8px; border-radius: 3px; text-decoration: underline;">December 2024 Report</a></p>
                
                <p>No immediate action required.</p>
                
                <p>SecureShield Data Protection</p>
            </div>""",
            "is_phishing": False,
            "order_id": 14,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "unknown_external",
            "urgency_level": "low",
            "framing_type": "threat",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "low", "framing": "threat"},
            "aggression_level": "low",
            "tactics": ["routine_monitoring", "service_notification"],
        },

        # EMAIL 15: LEGITIMATE | Unknown External | Low Urgency | Reward
        {
            "sender_name": "Cybersecurity Today",
            "sender_email": "newsletter@cybersecuritytoday.com",
            "subject": "This Week in Cybersecurity: New Ransomware Trends",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h3 style="margin-top: 0;">Top Stories This Week:</h3>
                
                <p><strong>Ransomware Trends:</strong> New variants targeting healthcare sector. <a href="https://cybersecuritytoday.com/articles/ransomware-2025" style="color: #0066cc; text-decoration: underline;">Read more</a></p>
                
                <p><strong>Best Practice Guide:</strong> Implementing zero-trust architecture. <a href="https://cybersecuritytoday.com/guides/zero-trust" style="color: #0066cc; text-decoration: underline;">Read more</a></p>
                
                <p><strong>Upcoming Webinar:</strong> "Phishing Defense Strategies" — Jan 25, 2 PM EST. <a href="https://cybersecuritytoday.com/webinars" style="color: #0066cc; text-decoration: underline;">Register</a></p>
                
                <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
                
                <p style="font-size: 12px; color: #666;">
                    <a href="https://cybersecuritytoday.com/unsubscribe" style="color: #666;">Unsubscribe</a> | 
                    <a href="https://cybersecuritytoday.com/preferences" style="color: #666;">Manage preferences</a>
                </p>
            </div>""",
            "is_phishing": False,
            "order_id": 15,
            "experimental": True,
            "email_type": "legitimate",
            "sender_familiarity": "unknown_external",
            "urgency_level": "low",
            "framing_type": "reward",
            "phishing_quality": None,
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": False,
            "requests_sensitive_info": False,
            "factorial_category": {"type": "legitimate", "sender": "unknown_external", "urgency": "low", "framing": "reward"},
            "aggression_level": "low",
            "tactics": ["informational_content", "newsletter"],
        },

        # EMAIL 16: PHISHING | Known Internal | Low Urgency | Reward
        {
            "sender_name": "ResearchConnect Study Team",
            "sender_email": "opportunities@researchconnect-premium-studies.com",
            "subject": "Thank You + More High-Paying Studies Available!",
            "body": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="background: #6d3ef2; color: white; padding: 15px; margin-bottom: 20px;">
                    <h2 style="margin: 0;">ResearchConnect</h2>
                </div>
                
                <p>Hi there,</p>
                
                <p>Thank you for completing the "Email Decision Making" study! Your payment has been processed.</p>
                
                <div style="background: #f0f7ff; padding: 15px; border-left: 4px solid #6d3ef2; margin: 16px 0;">
                    <p style="margin: 0;"><strong>Good news:</strong> You've been selected for our <strong>Premium Participant Program</strong> with higher-paying studies.</p>
                </div>
                
                <p>Exclusive opportunities available now:</p>
                
                <ul style="margin: 16px 0; padding-left: 24px;">
                    <li>Consumer Behavior Study — £18 for 40 minutes</li>
                    <li>Technology Survey — £15 for 30 minutes</li>
                    <li>Workplace Decision Making Part 2 — £22 for 45 minutes</li>
                </ul>
                
                <p><a href="http://researchconnect-premium-studies.com/claim" style="color: #0066cc; background-color: #e6f3ff; padding: 8px 16px; border-radius: 4px; text-decoration: underline; font-weight: bold;">View My Exclusive Studies</a></p>
                
                <p style="color: #d32f2f; font-size: 13px;"><em>These expire in 48 hours.</em></p>
            </div>""",
            "is_phishing": True,
            "order_id": 16,
            "experimental": True,
            "email_type": "phishing",
            "sender_familiarity": "known_internal",
            "urgency_level": "low",
            "framing_type": "reward",
            "phishing_quality": "high",
            "has_aggressive_content": False,
            "has_spelling_errors": False,
            "has_suspicious_url": True,
            "requests_sensitive_info": True,
            "factorial_category": {"type": "phishing", "sender": "known_internal", "urgency": "low", "framing": "reward"},
            "aggression_level": "medium",
            "tactics": ["contextual_trust_exploitation", "platform_impersonation", "exclusivity_scarcity"],
        }
    ]
    
    return emails


async def seed():
    """Seed the database with emails."""
    client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    
    print("Clearing existing data...")
    await db.emails.delete_many({})
    await db.participants.delete_many({})
    await db.logs.delete_many({})
    
    emails = get_seed_emails()
    
    result = await db.emails.insert_many(emails)
    print(f"Successfully inserted {len(result.inserted_ids)} emails")
    
    print("\n=== Factorial Design Summary ===")
    print(f"{'#':<3} {'Type':<11} {'Sender':<17} {'Urgency':<7} {'Framing':<7}")
    print("-" * 50)
    for email in emails:
        if email["experimental"]:
            print(f"#{email['order_id']:<2} {email['email_type']:<11} {email['sender_familiarity']:<17} {email['urgency_level']:<7} {email['framing_type']:<7}")
    
    # Verify factorial completeness
    print("\n=== Factorial Design Verification ===")
    combinations = set()
    for email in emails:
        if email["experimental"]:
            # Simplify sender to known/unknown
            sender = "known" if "known" in email["sender_familiarity"] else "unknown"
            combo = (email["email_type"], sender, email["urgency_level"], email["framing_type"])
            combinations.add(combo)
    
    expected = 16  # 2^4
    print(f"Unique combinations: {len(combinations)} / {expected}")
    if len(combinations) == expected:
        print("✅ All factorial combinations covered!")
    else:
        print("❌ Missing combinations!")


if __name__ == "__main__":
    asyncio.run(seed())