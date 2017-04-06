from incf.countryutils import transformations


def get_region(country):
    # TODO Map with more granularity. Especially countries in Asia that are
    # really middle eastern along with edge cases like South Africa

    cn_to_ctn = {
        'Antarctica': 'Other',
        'Bouvet Island': 'Other',
        'British Indian Ocean Territory': 'Other',
        'Congo, the Democratic Republic of the': 'Africa',
        "Cote D'Ivoire": 'Africa',xw
        'Heard Island and Mcdonald Islands': 'Oceania',
        'Iran, Islamic Republic of': 'Other',
        "Korea, Democratic People's Republic of": 'Other',
        'Korea, Republic of': 'Asia',
        'Kyrgyzstan': 'Asia',
        'Micronesia, Federated States of': 'Oceania',
        'Palestinian Territory, Occupied': 'Asia',
        'Pitcairn': 'Other',
        'Slovakia': 'Europe',
        'Svalbard and Jan Mayen': 'Europe',
        'Tanzania, United Republic of': 'Africa',
        'United Kingdom': 'Europe',
        'United States': 'North America',
        'Viet Nam': 'Asia',
        'Virgin Islands, British': 'North America',
        'Virgin Islands, U.s.': 'North America',
    }

    return cn_to_ctn[country] if country in cn_to_ctn else transformations.cn_to_ctn(country)