from flask import Flask, request, jsonify
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAssistant:
    def __init__(self, model):
        self.model = model
        self.inventory_data = None
        self.orders_data = None
        self.coupons_data = None
        self.product_embeddings = None
        self.REORDER_THRESHOLD = 20
        self.last_context = None

    # [Previous AIAssistant class methods remain exactly the same]
    # Copy all methods from the original AIAssistant class here

    def fetch_data(self):
        """Fetch and store all data from APIs"""
        try:
            requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
            
            # Fetch inventory data
            inventory_url = "http://10.123.79.112:1026/u/home/json/pmai006"
            inventory_payload = {"PMAI006Operation": {}}
            self.inventory_data = requests.post(inventory_url, json=inventory_payload, verify=False).json()
            
            # Fetch orders data
            orders_url = "http://10.123.79.112:1026/u/home/json/pmai009"
            orders_payload = {"PMAI009Operation": {}}
            self.orders_data = requests.post(orders_url, json=orders_payload, verify=False).json()
            
            # Fetch coupons data
            coupons_url = "http://10.123.79.112:1026/u/home/json/pmai016"
            coupons_payload = {"PMAI016Operation": {}}
            self.coupons_data = requests.post(coupons_url, json=coupons_payload, verify=False).json()
            
            self._generate_product_embeddings()
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise


    def get_best_investment_recommendations(self) -> str:
        """Generate best investment recommendations based on multiple factors"""
        products = self._get_products()
        sales_data = self._analyze_sales_trends()
        
        product_scores = []
        for product in products:
            score = self._calculate_investment_score(product, sales_data)
            product_scores.append((product, score))
        
        # Sort by score in descending order
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        response = "🎯 Top Investment Recommendations:\n\n"
        for i, (product, score) in enumerate(product_scores[:3], 1):
            response += (
                f"#{i} {product['ws_item_name']}\n"
                f"📊 Investment Score: {score:.1f}/100\n"
                f"💰 Current Price: ${float(product['ws_unit_price']):.2f}\n"
                f"📦 Stock Level: {product['ws_stock']}\n"
                f"📈 Sales Trend: {self._get_sales_trend(product['ws_item_id'], sales_data)}\n\n"
            )
        
        return response

    def check_inventory_status(self) -> str:
        """Generate comprehensive inventory status report"""
        inventory = self._analyze_inventory_health()
        
        response = "📦 Current Inventory Status:\n\n"
        
        # Low stock items
        if inventory['low_stock']:
            response += "⚠️ Low Stock Items:\n"
            for item in inventory['low_stock']:
                response += f"• {item['name']}: {item['stock']} units remaining\n"
            response += "\n"
        
        # Overstock items
        if inventory['overstock']:
            response += "📈 Overstocked Items:\n"
            for item in inventory['overstock']:
                response += f"• {item['name']}: {item['stock']} units (high inventory)\n"
            response += "\n"
        
        # Healthy stock items
        response += f"✅ {len(inventory['healthy_stock'])} items at healthy stock levels\n"
        
        return response

    def get_sales_performance(self) -> str:
        """Generate detailed sales performance analysis"""
        sales = self._analyze_sales_trends()
        
        response = "📊 Sales Performance Analysis:\n\n"
        
        # Overall metrics
        response += f"📈 Total Orders: {sales['total_orders']}\n\n"
        
        # Top selling products
        response += "🏆 Top Selling Products:\n"
        for prod_id, details in sales['popular_products']:
            response += (
                f"• {details['product_name']}\n"
                f"  📦 Units Sold: {details['quantity']}\n"
                f"  💰 Revenue: ${details['revenue']:,.2f}\n\n"
            )
        
        return response

    def analyze_coupons(self, query_type: str = 'all') -> str:
        """Generate coupon analysis based on query type with improved formatting"""
        coupon_data = self._analyze_coupon_effectiveness()
        
        if query_type == 'active':
            response = "🎟️ Active Coupons:\n\n"
            for coupon in coupon_data['active_coupons']:
                response += (
                    f"🏷️ {coupon['code']}\n"
                    f"   • Discount: {coupon['discount']}% off\n"
                    f"   • Campaign: {coupon['campaign']}\n"
                    f"   • Valid until: {coupon['end_date']}\n\n"
                )
            if not coupon_data['active_coupons']:
                response += "No active coupons found.\n"
        
        elif query_type == 'expired':
            response = "⏰ Expired Coupons:\n\n"
            for coupon in coupon_data['expired_coupons']:
                response += (
                    f"🏷️ {coupon['code']}\n"
                    f"   • Was: {coupon['discount']}% off\n"
                    f"   • Campaign: {coupon['campaign']}\n"
                    f"   • Expired: {coupon['end_date']}\n\n"
                )
            if not coupon_data['expired_coupons']:
                response += "No expired coupons found.\n"
        
        elif query_type == 'high_value':
            response = "💎 High-Value Coupons (>20% off):\n\n"
            for coupon in coupon_data['high_value_coupons']:
                status = "Active" if coupon in coupon_data['active_coupons'] else "Expired"
                response += (
                    f"🏷️ {coupon['code']}\n"
                    f"   • Discount: {coupon['discount']}% off\n"
                    f"   • Campaign: {coupon['campaign']}\n"
                    f"   • Status: {status}\n"
                    f"   • Valid until: {coupon['end_date']}\n\n"
                )
            if not coupon_data['high_value_coupons']:
                response += "No high-value coupons found.\n"
        
        else:
            response = "🎫 Coupon Overview:\n\n"
            response += f"✅ Active Coupons: {len(coupon_data['active_coupons'])}\n"
            response += f"⏰ Expired Coupons: {len(coupon_data['expired_coupons'])}\n"
            response += f"💎 High-Value Coupons: {len(coupon_data['high_value_coupons'])}\n"
        
        return response

    def _calculate_investment_score(self, product: Dict, sales_data: Dict) -> float:
        """Calculate investment score for a product based on multiple factors"""
        product_id = product['ws_item_id']
        product_sales = sales_data['product_sales'].get(product_id, {'quantity': 0, 'revenue': 0})
        
        # Calculate various metrics
        stock_level = float(product['ws_stock'])
        unit_price = float(product['ws_unit_price'])
        cost_price = float(product.get('ws_cost_price', 0))
        
        # Sales velocity (30% weight)
        sales_velocity = product_sales['quantity'] / 30
        velocity_score = min(sales_velocity * 10, 30)
        
        # Profit margin (25% weight)
        margin = ((unit_price - cost_price) / unit_price) * 100
        margin_score = min(margin, 25)
        
        # Stock efficiency (25% weight)
        ideal_stock = sales_velocity * 14  # 2 weeks of stock
        efficiency = 25 * (1 - abs(stock_level - ideal_stock) / ideal_stock)
        efficiency_score = max(0, efficiency)
        
        # Revenue contribution (20% weight)
        total_revenue = sum(p['revenue'] for p in sales_data['product_sales'].values())
        revenue_score = (product_sales['revenue'] / total_revenue * 100) if total_revenue > 0 else 0
        revenue_score = min(revenue_score * 2, 20)
        
        return velocity_score + margin_score + efficiency_score + revenue_score

    def _get_sales_trend(self, product_id: str, sales_data: Dict) -> str:
        """Calculate sales trend for a product"""
        if product_id not in sales_data['product_sales']:
            return "No sales data"
        
        sales = sales_data['product_sales'][product_id]['quantity']
        if sales == 0:
            return "No recent sales"
        elif sales > 100:
            return "High demand"
        elif sales > 50:
            return "Moderate demand"
        else:
            return "Low demand"

    def _get_products(self) -> List[Dict]:
        """Get products from inventory data"""
        return (self.inventory_data.get('PMAI006OperationResponse', {})
                .get('ws_invent_recout', {})
                .get('ws_invent_res', []))

    def _generate_product_embeddings(self):
        """Generate embeddings for product descriptions"""
        self.products = (self.inventory_data.get('PMAI006OperationResponse', {})
                        .get('ws_invent_recout', {})
                        .get('ws_invent_res', []))
        
        product_descriptions = []
        for product in self.products:
            description = (f"{product.get('ws_item_name', '')} - "
                         f"{product.get('ws_description', '')} - "
                         f"Category: {product.get('ws_category', '')}")
            product_descriptions.append(description)
        
        self.product_embeddings = self.model.encode(product_descriptions)

    def _analyze_inventory_health(self) -> Dict[str, List[Dict]]:
        """Analyze inventory health status"""
        inventory_health = {
            'low_stock': [],
            'healthy_stock': [],
            'overstock': []
        }

        products = (self.inventory_data.get('PMAI006OperationResponse', {})
                   .get('ws_invent_recout', {})
                   .get('ws_invent_res', []))

        for product in products:
            stock = int(product.get('ws_stock', 0))
            product_info = {
                'id': product.get('ws_item_id'),
                'name': product.get('ws_item_name'),
                'stock': stock,
                'price': float(product.get('ws_unit_price', 0))
            }

            if stock <= self.REORDER_THRESHOLD:
                inventory_health['low_stock'].append(product_info)
            elif stock > 50:  # Overstock threshold
                inventory_health['overstock'].append(product_info)
            else:
                inventory_health['healthy_stock'].append(product_info)

        return inventory_health

    def _analyze_sales_trends(self) -> Dict[str, Any]:
        """Analyze sales trends from orders data"""
        sales_analysis = {
            'total_orders': 0,
            'product_sales': {},
            'popular_products': []
        }

        orders = (self.orders_data.get('PMAI009OperationResponse', {})
                 .get('ws_order_recout', {})
                 .get('ws_order_res', []))
        
        sales_analysis['total_orders'] = len(orders)

        for order in orders:
            prod_id = order.get('ws_item_id')
            if prod_id not in sales_analysis['product_sales']:
                sales_analysis['product_sales'][prod_id] = {
                    'quantity': 0,
                    'revenue': 0,
                    'product_name': order.get('ws_item_name')
                }
            
            quantity = int(order.get('ws_quantity', 0))
            price = float(order.get('ws_unit_price', 0))
            sales_analysis['product_sales'][prod_id]['quantity'] += quantity
            sales_analysis['product_sales'][prod_id]['revenue'] += quantity * price

        # Sort products by quantity sold
        popular_products = sorted(
            sales_analysis['product_sales'].items(),
            key=lambda x: x[1]['quantity'],
            reverse=True
        )
        sales_analysis['popular_products'] = popular_products[:3]

        return sales_analysis

    def _analyze_coupon_effectiveness(self) -> Dict[str, List[Dict]]:
        """Analyze coupon effectiveness with correct JSON structure"""
        current_date = datetime.now().date()
        coupon_analysis = {
            'active_coupons': [],
            'expired_coupons': [],
            'high_value_coupons': []
        }

        coupons = (self.coupons_data.get('PMAI016OperationResponse', {})
                .get('ws_coupon_recout', {})
                .get('ws_coupon_res', []))

        for coupon in coupons:
            start_date = datetime.strptime(coupon.get('ws_start_date', ''), '%Y-%m-%d').date()
            end_date = datetime.strptime(coupon.get('ws_end_date', ''), '%Y-%m-%d').date()
            offer_percent = float(coupon.get('ws_offer_percent', 0))
            
            coupon_info = {
                'code': coupon.get('ws_coupon_code'),
                'discount': offer_percent,
                'campaign': coupon.get('ws_campaigns_name'),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }

            if current_date <= end_date:
                coupon_analysis['active_coupons'].append(coupon_info)
            else:
                coupon_analysis['expired_coupons'].append(coupon_info)
                
            if offer_percent > 20:  # High value threshold
                coupon_analysis['high_value_coupons'].append(coupon_info)

        return coupon_analysis  
    def generate_response(self, query: str, context: Dict[str, Any]) -> str:
        """Generate natural language response based on context"""
        # Investment recommendations
        if 'investment' in query.lower():
            if 'best_product' in context:
                product = context['best_product']
                return (f"💡 Investment Recommendation:\n\n"
                       f"🏆 Top Pick: {product['ws_item_name']}\n"
                       f"📊 Analysis:\n"
                       f"- 📦 Current stock: {product['ws_stock']} units\n"
                       f"- 💰 Price point: ${product['ws_unit_price']}\n"
                       f"- 🏷️ Category: {product['ws_category']}\n"
                       f"- ✨ Features: {product['ws_description']}\n\n"
                       f"Would you like to see detailed sales analytics for this product? 📈")

        # Inventory and reorder queries
        if any(word in query.lower() for word in ['stock', 'inventory', 'reorder']):
            inventory = context.get('inventory_status', {})
            low_stock = inventory.get('low_stock', [])
            if low_stock:
                response = "🚨 Low Stock Alert!\n\n"
                for product in low_stock:
                    response += (f"📉 {product['name']}\n"
                               f"   • Current stock: {product['stock']} units\n"
                               f"   • Threshold: {self.REORDER_THRESHOLD} units\n")
                response += "\n⚡ Recommendation: Place reorder requests for these items soon.\n"
                response += "Need help calculating optimal reorder quantities? 🤔"
                return response

        # Coupon-related queries
        if 'coupon' in query.lower():
            coupon_status = context.get('coupon_status', {})
            if 'expired' in query.lower():
                expired = coupon_status.get('expired_coupons', [])
                if expired:
                    response = "⏰ Expired Coupons:\n\n"
                    for coupon in expired:
                        response += f"🏷️ {coupon['code']}: {coupon['discount']}% off (Expired: {coupon['end_date']})\n"
                    return response
                return "✨ No expired coupons found in the system."
            elif 'active' in query.lower():
                active = coupon_status.get('active_coupons', [])
                if active:
                    response = "✨ Active Coupon Codes:\n\n"
                    for coupon in active:
                        response += f"🎟️ {coupon['code']}: {coupon['discount']}% off (Expires: {coupon['end_date']})\n"
                    return response
                return "😔 No active coupons found in the system."
            elif 'suggest' in query.lower() or 'recommend' in query.lower():
                overstock = context.get('inventory_status', {}).get('overstock', [])
                if overstock:
                    response = "🎯 Recommended Promotions:\n\n"
                    for product in overstock[:3]:
                        suggested_discount = min(30, int(float(product['price']) * 0.15))
                        response += f"📦 {product['name']}\n"
                        response += f"   • Suggested discount: {suggested_discount}%\n"
                        response += f"   • Current price: ${product['price']}\n"
                    return response

        # Sales analysis
        if 'sales' in query.lower():
            sales = context.get('sales_analysis', {})
            if sales:
                popular = sales.get('popular_products', [])
                response = "📊 Sales Performance Summary:\n\n"
                response += f"📈 Total orders: {sales.get('total_orders', 0)}\n\n"
                if popular:
                    response += "🏆 Top Selling Products:\n"
                    for prod_id, details in popular:
                        response += f"✨ {details['product_name']}\n"
                        response += f"   • Units sold: {details['quantity']}\n"
                        response += f"   • Revenue: ${details['revenue']:,.2f}\n"
                return response

        return ("👋 Hello! I'm your retail assistant. I can help you with:\n\n"
                "📈 Product investment recommendations\n"
                "📦 Inventory management\n"
                "🏷️ Coupon analysis\n"
                "📊 Sales insights\n\n"
                "What would you like to know? 😊")

    def _analyze_product_potential(self) -> Dict[str, Any]:
        """Analyze product investment potential based on multiple factors"""
        product_analysis = {}
        
        # Get sales data
        sales_data = self._analyze_sales_trends()
        inventory_health = self._analyze_inventory_health()
        
        # Analyze each product
        for product in self.products:
            product_id = product.get('ws_item_id')
            
            # Calculate key metrics
            sales_info = sales_data['product_sales'].get(product_id, {
                'quantity': 0,
                'revenue': 0
            })
            
            current_stock = int(product.get('ws_stock', 0))
            unit_price = float(product.get('ws_unit_price', 0))
            
            # Calculate investment score based on multiple factors
            investment_score = {
                'sales_velocity': sales_info['quantity'] / 30 if sales_info['quantity'] > 0 else 0,  # Units sold per day
                'profit_margin': (unit_price - float(product.get('ws_cost_price', 0))) / unit_price * 100,
                'stock_efficiency': min(current_stock / max(sales_info['quantity'], 1), 1),  # Stock turnover ratio
                'revenue_contribution': sales_info['revenue'] / max(sum(p['revenue'] for p in sales_data['product_sales'].values()), 1) * 100
            }
            
            # Calculate weighted score
            weights = {
                'sales_velocity': 0.35,
                'profit_margin': 0.25,
                'stock_efficiency': 0.20,
                'revenue_contribution': 0.20
            }
            
            total_score = sum(score * weights[metric] for metric, score in investment_score.items())
            
            product_analysis[product_id] = {
                'product_info': product,
                'metrics': investment_score,
                'total_score': total_score,
                'recommendation_factors': []
            }
            
            # Add recommendation factors
            if investment_score['sales_velocity'] > 1:
                product_analysis[product_id]['recommendation_factors'].append('High sales velocity')
            if investment_score['profit_margin'] > 30:
                product_analysis[product_id]['recommendation_factors'].append('Strong profit margin')
            if investment_score['revenue_contribution'] > 10:
                product_analysis[product_id]['recommendation_factors'].append('Significant revenue contributor')
            
        return product_analysis

    def generate_investment_response(self, product_analysis: Dict[str, Any]) -> str:
        """Generate detailed investment recommendation response"""
        # Sort products by total score
        sorted_products = sorted(
            product_analysis.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Get top 3 recommendations
        top_recommendations = sorted_products[:3]
        
        response = "🎯 Investment Recommendations:\n\n"
        
        for rank, (prod_id, analysis) in enumerate(top_recommendations, 1):
            product = analysis['product_info']
            metrics = analysis['metrics']
            
            response += f"#{rank} - {product.get('ws_item_name')}\n"
            response += f"📊 Investment Score: {analysis['total_score']:.1f}/100\n"
            response += f"💰 Price: ${float(product.get('ws_unit_price', 0)):.2f}\n"
            response += f"📈 Sales Velocity: {metrics['sales_velocity']:.1f} units/day\n"
            response += f"✨ Profit Margin: {metrics['profit_margin']:.1f}%\n"
            
            if analysis['recommendation_factors']:
                response += "🌟 Key Strengths:\n"
                for factor in analysis['recommendation_factors']:
                    response += f"   • {factor}\n"
            
            response += "\n"
        
        response += "Would you like detailed analytics for any of these products? 📊"
        return response

    def _check_reorder_levels(self) -> Dict[str, Any]:
        """Analyze products that need restocking based on minimum reorder levels"""
        reorder_analysis = {
            'urgent_reorder': [],
            'approaching_reorder': [],
            'reorder_suggestions': {}
        }
        
        products = (self.inventory_data.get('PMAI006OperationResponse', {})
                .get('ws_invent_recout', {})
                .get('ws_invent_res', []))
        
        # Get sales data for calculating reorder quantities
        sales_data = self._analyze_sales_trends()
        
        for product in products:
            current_stock = int(product.get('ws_stock', 0))
            product_id = product.get('ws_item_id')
            min_stock = int(product.get('ws_min_stock', self.REORDER_THRESHOLD))
            
            # Calculate average daily sales
            product_sales = sales_data.get('product_sales', {}).get(product_id, {})
            sales_quantity = product_sales.get('quantity', 0)
            avg_daily_sales = sales_quantity / 30 if sales_quantity > 0 else 0
            
            # Calculate days of inventory remaining
            days_remaining = float('inf') if avg_daily_sales == 0 else current_stock / avg_daily_sales
            
            product_info = {
                'id': product_id,
                'name': product.get('ws_item_name'),
                'current_stock': current_stock,
                'min_stock': min_stock,
                'avg_daily_sales': avg_daily_sales,
                'days_remaining': days_remaining,
                'category': product.get('ws_category'),
                'unit_price': float(product.get('ws_unit_price', 0))
            }
            
            # Calculate suggested reorder quantity
            lead_time_days = 7  # Assumed lead time, adjust as needed
            safety_stock = min_stock * 0.5  # 50% of min stock as safety stock
            reorder_quantity = int(
                (avg_daily_sales * lead_time_days) + safety_stock - current_stock
            )
            
            if current_stock <= min_stock:
                product_info['urgency'] = 'URGENT'
                product_info['reorder_quantity'] = max(reorder_quantity, min_stock)
                reorder_analysis['urgent_reorder'].append(product_info)
            elif current_stock <= (min_stock * 1.5):
                product_info['urgency'] = 'SOON'
                product_info['reorder_quantity'] = max(reorder_quantity, 0)
                reorder_analysis['approaching_reorder'].append(product_info)
                
            if reorder_quantity > 0:
                reorder_analysis['reorder_suggestions'][product_id] = {
                    'quantity': reorder_quantity,
                    'reason': f"Based on {avg_daily_sales:.1f} units/day average sales and {lead_time_days} days lead time"
                }
        
        return reorder_analysis

    def generate_reorder_response(self, reorder_analysis: Dict[str, Any]) -> str:
        """Generate detailed response for reorder level analysis"""
        response = "📦 Inventory Reorder Analysis:\n\n"
        
        # Handle urgent reorders
        if reorder_analysis['urgent_reorder']:
            response += "🚨 URGENT REORDER REQUIRED:\n"
            for product in reorder_analysis['urgent_reorder']:
                response += (
                    f"• {product['name']}\n"
                    f"  📊 Current Stock: {product['current_stock']} units\n"
                    f"  ⚠️ Minimum Level: {product['min_stock']} units\n"
                    f"  📈 Avg Daily Sales: {product['avg_daily_sales']:.1f} units\n"
                    f"  ⏳ Days of Stock Left: {product['days_remaining']:.1f} days\n"
                    f"  🎯 Suggested Reorder: {product['reorder_quantity']} units\n\n"
                )
        
        # Handle approaching reorder level
        if reorder_analysis['approaching_reorder']:
            response += "⚠️ APPROACHING REORDER LEVEL:\n"
            for product in reorder_analysis['approaching_reorder']:
                response += (
                    f"• {product['name']}\n"
                    f"  📊 Current Stock: {product['current_stock']} units\n"
                    f"  ⚠️ Minimum Level: {product['min_stock']} units\n"
                    f"  📈 Avg Daily Sales: {product['avg_daily_sales']:.1f} units\n"
                    f"  ⏳ Days of Stock Left: {product['days_remaining']:.1f} days\n"
                    f"  🎯 Suggested Reorder: {product['reorder_quantity']} units\n\n"
                )
        
        if not (reorder_analysis['urgent_reorder'] or reorder_analysis['approaching_reorder']):
            response += "✅ All products are above minimum reorder levels.\n\n"
        
        response += (
            "📝 Note: Reorder quantities are calculated based on:\n"
            "• Average daily sales\n"
            "• Lead time (7 days)\n"
            "• Safety stock (50% of minimum stock)\n\n"
            "Would you like detailed analytics for any specific product? 🔍"
        )
        
        return response
    
    def process_query(self, query: str) -> str:
        """Enhanced query processing with better query type detection"""
        try:
            self.fetch_data()
            
            # Identify query type using keyword matching
            reorder_keywords = ['reorder', 'stock', 'inventory', 'level', 'minimum', 'refill', 'replenish']
            investment_keywords = ['invest', 'buy', 'purchase', 'recommend', 'best product']
            
            query_lower = query.lower()
            
            # Handle reorder level queries
            if any(keyword in query_lower for keyword in reorder_keywords):
                reorder_analysis = self._check_reorder_levels()
                return self.generate_reorder_response(reorder_analysis)
                
            # Handle investment recommendation queries
            elif any(keyword in query_lower for keyword in investment_keywords):
                product_analysis = self._analyze_product_potential()
                return self.generate_investment_response(product_analysis)
                
            # Default context-based response
            context = {
                'inventory_status': self._analyze_inventory_health(),
                'sales_analysis': self._analyze_sales_trends(),
                'coupon_status': self._analyze_coupon_effectiveness()
            }
            return self.generate_response(query, context)

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return ("😔 I apologize, but I encountered an error while processing your request. "
                "Could you please rephrase your question? 🤔")

# Initialize Flask app
app = Flask(__name__)

# Initialize the AI Assistant
try:
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    assistant = AIAssistant(model)
except Exception as e:
    logger.error(f"Error initializing AI Assistant: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    """Handle incoming queries"""
    try:
        # Get request data
        data = request.get_json()
        
        # Validate request
        if not data or 'query' not in data:
            return jsonify({
                'error': 'Invalid request. Please provide a query field.',
                'example': {
                    'query': 'Show me inventory status'
                }
            }), 400
            
        query = data['query']
        
        # Process query
        response = assistant.process_query(query)
        
        # Return response
        return jsonify({
            'status': 'success',
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)