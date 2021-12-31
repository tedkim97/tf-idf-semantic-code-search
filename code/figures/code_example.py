

def calculate_cart_taxes(cart, coupons, customerID, region):
    '''
    Calculates the taxes within a cart depending on configuration
    '''
    DB_URL = '...' # TODO: Remove hardcoded url

    # some items are tax exempt depending on the location of the customer
    tax_rate, exceptions = RetrieveTaxDetails(DB_URL, region)
    if not validateCoupons(customerID, coupons):
        raise ValueError    

    discounted_cart = applyCoupons(taxable_cart_amount, cart)
    
    # TODO: Use the new cart calculating library
    taxable_cart_amount = 0
    for product in discounted_cart:
        if not exception[product]:
            taxable_cart_amount += product.price
    
    return total_tax


"""
"calculate_cart_taxes cart coupons customerid region calculates the taxes within
a cart depending on configuration db_url todo remove hardcoded url some
items are tax exempt depending on the location of the customer tax_rate exceptions
retrievetaxdetails db_url region validatecoupons customerid coupons discounted_cart..."

applyCoupons
taxable_cart_amount
cart
    



# Tokenization W/out Filtering
Same as above but also include 
+=
valueerror
raise
for
in
not

"""



def CalculateCartTaxes(cart, customerID, region):
    '''
    Calculates taxes depending on configuration
    '''
    DB_URL = '...' # TODO: Remove hardcoded url
    tax_rate, exceptions = RetrieveTaxDetails(DB_URL, region)
    return sum(cart) * tax_rate

'''
"def CalculateCartTaxes cart customerID region Calculates taxes depending on configuration DB_URL = '...' TODO: Remove hardcoded url tax_rate exceptions = RetrieveTaxDetails DB_URL region return sum(cart) * tax_rate"
'''

'''
def 
CalculateCartTaxes
cart
customerID
region
Calculates
taxes
depending
on
configuration
DB_URL
= 
'...'
TODO:
Remove
hardcoded
url
tax_rate
exceptions
=
RetrieveTaxDetails
DB_URL
region
return
sum(cart)
*
tax_rate
'''

def CalculateCartTaxes(cart, customerID, region):
    '''
    Calculates taxes depending on configuration
    '''
    DB_URL = '...' # TODO: Remove hardcoded url
    tax_rate, exceptions = RetrieveTaxDetails(DB_URL, region)
    return sum(cart) * tax_rate


def upload_to_db(conn_url, index, key, detail):
    '''
    Update the key on index with {detail}
    '''
    with (db_conn(conn_url)) as db:
        temp_index = db.get_index(index)
        temp_index.update(key, detail)

    return True

def convert_int_to_string(number): 
    return str(number)

def salt_hash_password(string_id, password):
    salted_pw = salt_and_pepper(password)
    hashed_pw = cryptolib.SHA256(password)
    try:
        CREDENTIAL_DB.add(string_id, hashed_pw)
    except:
        return False
    return True
